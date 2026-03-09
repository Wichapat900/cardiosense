"""
train.py — CardioSense Training Pipeline (Explainable RF Edition)
================================================================
Trains a Random Forest on an extended HRV feature set.
Every decision is fully explainable via SHAP values.

Features (20 total):
  Time-domain:   mean_rr, sdnn, rmssd, pnn50, cv, mean_hr
  Poincaré:      sd1, sd2, sd1_sd2_ratio
  Nonlinear:     sample_entropy, approx_entropy, turning_point_ratio
  Frequency:     dominant_freq, spectral_entropy
  Morphology:    p_wave_absence, qrs_width, t_wave_ratio
  Distribution:  rr_skewness, rr_kurtosis, rr_range

Usage:
  python src/train.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SAMPLE_RATE = 250
RESULTS_DIR = Path("models/results")
SAVED_DIR   = Path("models/saved")
DATA_DIR    = Path("data/processed")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_DIR.mkdir(parents=True, exist_ok=True)


# ─── Extended HRV Feature Extraction ──────────────────────────────────────────

def bandpass(signal, fs=SAMPLE_RATE, lo=0.5, hi=40.0):
    nyq = fs / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_rpeaks(signal, fs=SAMPLE_RATE):
    sig = bandpass(signal, fs)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    if np.abs(sig.min()) > np.abs(sig.max()):
        sig = -sig
    sig_max = float(np.max(sig))
    thr = max(0.35, sig_max * 0.35)
    peaks, _ = find_peaks(sig, height=thr, distance=int(0.32 * fs),
                          prominence=sig_max * 0.28, wlen=int(0.6 * fs))
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, height=max(0.25, sig_max * 0.25),
                              distance=int(0.32 * fs), prominence=sig_max * 0.20)
    return peaks, sig


def sample_entropy(rr, m=2, r_factor=0.2):
    """Sample entropy of RR series — measures unpredictability."""
    if len(rr) < 10:
        return 0.0
    r = r_factor * np.std(rr)
    N = len(rr)
    def _count(template_len):
        count = 0
        for i in range(N - template_len):
            template = rr[i:i + template_len]
            for j in range(i + 1, N - template_len + 1):
                if np.max(np.abs(rr[j:j + template_len] - template)) < r:
                    count += 1
        return count
    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return 0.0
    return -np.log(A / B + 1e-10)


def approx_entropy(rr, m=2, r_factor=0.2):
    """Approximate entropy — similar to SampEn but includes self-matches."""
    if len(rr) < 10:
        return 0.0
    r = r_factor * np.std(rr)
    N = len(rr)
    def _phi(template_len):
        count = np.zeros(N - template_len + 1)
        for i in range(N - template_len + 1):
            template = rr[i:i + template_len]
            for j in range(N - template_len + 1):
                if np.max(np.abs(rr[j:j + template_len] - template)) <= r:
                    count[i] += 1
        return np.mean(np.log(count / (N - template_len + 1) + 1e-10))
    return abs(_phi(m) - _phi(m + 1))


def turning_point_ratio(rr):
    """Fraction of RR intervals that are turning points (local min/max).
    AFib has much higher TPR than normal sinus rhythm."""
    if len(rr) < 3:
        return 0.0
    arr = np.array(rr)
    tp = np.sum(
        ((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])) |
        ((arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:]))
    )
    return tp / (len(arr) - 2)


def dominant_frequency(rr, fs_rr=4.0):
    """Dominant frequency in RR spectrum.
    Normal: clear peak at ~0.1Hz (LF) or ~0.25Hz (HF).
    AFib: flat/noisy spectrum, low dominant frequency power."""
    if len(rr) < 8:
        return 0.0, 0.0
    from scipy.interpolate import interp1d
    t_rr = np.cumsum(rr) / 1000.0
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_rr)
    if len(t_uniform) < 4:
        return 0.0, 0.0
    try:
        f_interp = interp1d(t_rr, rr, kind="linear", bounds_error=False,
                            fill_value="extrapolate")
        rr_uniform = f_interp(t_uniform)
        freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=min(len(rr_uniform), 64))
        dom_idx = np.argmax(psd)
        dom_freq = float(freqs[dom_idx])
        # Spectral entropy — flat spectrum = high entropy = AFib
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))
        return dom_freq, spec_ent
    except Exception:
        return 0.0, 0.0


def p_wave_absence_score(signal, peaks, fs=SAMPLE_RATE):
    """Estimate P-wave absence — key morphological AFib marker.
    Looks at the region 200-100ms before each R-peak for P-wave energy.
    Low P-wave energy = AFib."""
    if len(peaks) < 3:
        return 0.5
    scores = []
    for p in peaks:
        p_start = p - int(0.20 * fs)
        p_end   = p - int(0.05 * fs)
        if p_start < 0:
            continue
        p_region = signal[p_start:p_end]
        qrs_region = signal[p - int(0.05 * fs): p + int(0.05 * fs)]
        if len(p_region) == 0 or len(qrs_region) == 0:
            continue
        p_energy   = float(np.mean(p_region ** 2))
        qrs_energy = float(np.mean(qrs_region ** 2))
        # Low ratio = P-wave absent relative to QRS
        scores.append(p_energy / (qrs_energy + 1e-8))
    return float(np.mean(scores)) if scores else 0.5


def extract_features(signal, fs=SAMPLE_RATE):
    """Extract all 20 HRV + morphological features."""
    peaks, sig_norm = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000  # ms
    rr = rr[(rr > 300) & (rr < 2000)]

    if len(rr) < 4:
        return None  # not enough beats

    # ── Time domain ───────────────────────────────────────────────────────────
    mean_rr  = float(np.mean(rr))
    sdnn     = float(np.std(rr))
    rmssd    = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    pnn50    = float(np.mean(np.abs(np.diff(rr)) > 50))
    cv       = sdnn / mean_rr if mean_rr > 0 else 0.0
    mean_hr  = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    # ── Poincaré ──────────────────────────────────────────────────────────────
    sd1 = float(np.std(np.diff(rr)) / np.sqrt(2))
    sd2 = float(np.sqrt(max(0, 2 * sdnn ** 2 - 0.5 * np.std(np.diff(rr)) ** 2)))
    sd1_sd2_ratio = sd1 / (sd2 + 1e-8)

    # ── Nonlinear ─────────────────────────────────────────────────────────────
    samp_ent = sample_entropy(rr)
    appr_ent = approx_entropy(rr)
    tpr      = turning_point_ratio(rr)

    # ── Frequency domain ──────────────────────────────────────────────────────
    dom_freq, spec_ent = dominant_frequency(rr)

    # ── Morphological ─────────────────────────────────────────────────────────
    p_absent = p_wave_absence_score(sig_norm, peaks, fs)

    # QRS width estimate — average width of QRS complex
    qrs_widths = []
    for p in peaks:
        left  = p - int(0.05 * fs)
        right = p + int(0.05 * fs)
        if left >= 0 and right < len(sig_norm):
            seg = sig_norm[left:right]
            above = np.where(seg > 0.5 * sig_norm[p])[0]
            qrs_widths.append(len(above) / fs * 1000 if len(above) > 0 else 80)
    qrs_width = float(np.mean(qrs_widths)) if qrs_widths else 80.0

    # T-wave ratio — T-wave energy relative to QRS
    t_ratios = []
    for p in peaks:
        t_start = p + int(0.15 * fs)
        t_end   = p + int(0.40 * fs)
        if t_end < len(sig_norm):
            t_e   = float(np.mean(sig_norm[t_start:t_end] ** 2))
            qrs_e = float(np.mean(sig_norm[p - int(0.05*fs): p + int(0.05*fs)] ** 2))
            t_ratios.append(t_e / (qrs_e + 1e-8))
    t_wave_ratio = float(np.mean(t_ratios)) if t_ratios else 0.1

    # ── Distribution ──────────────────────────────────────────────────────────
    rr_skew = float(skew(rr))
    rr_kurt = float(kurtosis(rr))
    rr_range = float(np.max(rr) - np.min(rr))

    return {
        # Time domain
        "mean_rr":        mean_rr,
        "sdnn":           sdnn,
        "rmssd":          rmssd,
        "pnn50":          pnn50,
        "cv":             cv,
        "mean_hr":        mean_hr,
        # Poincaré
        "sd1":            sd1,
        "sd2":            sd2,
        "sd1_sd2_ratio":  sd1_sd2_ratio,
        # Nonlinear
        "sample_entropy": samp_ent,
        "approx_entropy": appr_ent,
        "turning_point_ratio": tpr,
        # Frequency
        "dominant_freq":  dom_freq,
        "spectral_entropy": spec_ent,
        # Morphological
        "p_wave_absence": p_absent,
        "qrs_width":      qrs_width,
        "t_wave_ratio":   t_wave_ratio,
        # Distribution
        "rr_skewness":    rr_skew,
        "rr_kurtosis":    rr_kurt,
        "rr_range":       rr_range,
    }


FEATURE_NAMES = [
    "mean_rr", "sdnn", "rmssd", "pnn50", "cv", "mean_hr",
    "sd1", "sd2", "sd1_sd2_ratio",
    "sample_entropy", "approx_entropy", "turning_point_ratio",
    "dominant_freq", "spectral_entropy",
    "p_wave_absence", "qrs_width", "t_wave_ratio",
    "rr_skewness", "rr_kurtosis", "rr_range",
]


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data():
    signals = np.load(DATA_DIR / "signals.npy")
    labels  = np.load(DATA_DIR / "labels.npy")

    # Try to load patient IDs for group-aware splitting
    pid_path = DATA_DIR / "patient_ids.npy"
    patient_ids = np.load(pid_path) if pid_path.exists() else np.arange(len(signals))

    print(f"Loaded {len(signals)} segments — {np.sum(labels==1)} AFib, {np.sum(labels==0)} Normal")

    print("Extracting extended HRV features...")
    X, y, groups = [], [], []
    for i, (sig, lbl, pid) in enumerate(zip(signals, labels, patient_ids)):
        if i % 500 == 0:
            print(f"  {i}/{len(signals)}")
        feats = extract_features(sig, SAMPLE_RATE)
        if feats is None:
            continue
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(int(lbl))
        groups.append(int(pid))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=int)
    print(f"Features extracted: {X.shape}  AFib={np.sum(y==1)}  Normal={np.sum(y==0)}")
    return X, y, groups


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    X, y, groups = load_data()

    # Patient-wise stratified split (80/10/10)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # SMOTE oversampling on training set only
    sm = SMOTE(sampling_strategy=0.8, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_res)}  AFib={np.sum(y_res==1)}  Normal={np.sum(y_res==0)}")

    # Random Forest with class weighting
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 2.5},   # penalise missed AFib
        n_jobs=-1,
        random_state=42,
    )
    scaler = StandardScaler()
    X_res_s  = scaler.fit_transform(X_res)
    X_val_s  = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    rf.fit(X_res_s, y_res)

    # ── Optimise threshold on validation set ──────────────────────────────────
    val_probs = rf.predict_proba(X_val_s)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    from sklearn.metrics import f1_score
    for t in np.arange(0.20, 0.80, 0.01):
        preds = (val_probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    print(f"Optimal threshold: {best_thresh:.2f}  (val F1={best_f1:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_probs = rf.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    report = {
        "model":          "Random Forest (Explainable)",
        "n_features":     len(FEATURE_NAMES),
        "feature_names":  FEATURE_NAMES,
        "threshold":      float(best_thresh),
        "roc_auc":        float(roc_auc_score(y_test, test_probs)),
        "avg_precision":  float(average_precision_score(y_test, test_probs)),
        "sensitivity":    float(tp / (tp + fn + 1e-8)),
        "specificity":    float(tn / (tn + fp + 1e-8)),
        "f1":             float(f1_score(y_test, test_preds, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    print(f"\n{'='*50}")
    print(f"  AUC-ROC:     {report['roc_auc']:.4f}")
    print(f"  Sensitivity: {report['sensitivity']:.4f}  (AFib recall)")
    print(f"  Specificity: {report['specificity']:.4f}")
    print(f"  F1:          {report['f1']:.4f}")
    print(f"{'='*50}")

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # ── Save artifacts ────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "report_rf.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save scaler + model + threshold together
    bundle = {"model": rf, "scaler": scaler,
              "threshold": best_thresh, "feature_names": FEATURE_NAMES}
    joblib.dump(bundle, SAVED_DIR / "rf_pipeline.pkl")
    print(f"\nSaved → models/saved/rf_pipeline.pkl")
    print(f"Saved → models/results/report_rf.json")
    print(f"Saved → models/results/feature_importance.csv")

    return bundle


if __name__ == "__main__":
    train()