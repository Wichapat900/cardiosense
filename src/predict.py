"""
predict.py — CardioSense Inference Module
==========================================
Used by app.py to run inference on:
  - Uploaded .csv ECG files
  - Live AD8232 serial data (via buffer)
  - Synthetic demo signals

Exposes a single predict() function that returns:
  {
    "afib_probability": float,      # 0–1
    "classification":   str,        # "Normal" | "Borderline" | "AFib"
    "confidence":       str,        # "Low" | "Medium" | "High"
    "hrv_features":     dict,       # RMSSD, SDNN, etc.
    "rr_intervals":     list[float],
  }
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR    = Path("models/saved")
PROCESSED_DIR = Path("data/processed")
SAMPLE_RATE   = 250
SEGMENT_LEN   = 250 * 30   # 30s @ 250Hz

# Lazy-loaded model cache
_rf_pipeline = None
_cnn_model   = None
_meta        = None


def _load_meta():
    global _meta
    if _meta is None:
        meta_path = PROCESSED_DIR / "meta.pkl"
        if meta_path.exists():
            _meta = joblib.load(meta_path)
        else:
            _meta = {
                "feature_cols": [
                    "mean_rr", "sdnn", "rmssd", "pnn50", "cv", "mean_hr",
                    "sd1", "sd2", "sd_ratio", "sample_ent", "hist_ent",
                    "prop_short", "prop_long", "nn50", "n_beats"
                ]
            }
    return _meta


def _load_rf():
    global _rf_pipeline
    if _rf_pipeline is None:
        path = MODELS_DIR / "rf_pipeline.pkl"
        if path.exists():
            _rf_pipeline = joblib.load(path)
    return _rf_pipeline


def _load_cnn():
    global _cnn_model
    if _cnn_model is None:
        try:
            import torch
            from train import AFibCNNBiLSTM
            path = MODELS_DIR / "cnn_best.pt"
            if path.exists():
                device = torch.device("cpu")
                ckpt = torch.load(path, map_location=device)
                model = AFibCNNBiLSTM()
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _cnn_model = model
        except Exception:
            pass
    return _cnn_model


# ─── Signal Processing ────────────────────────────────────────────────────────

def normalize_polarity(signal: np.ndarray) -> np.ndarray:
    """
    Automatic polarity normalization — standard in all clinical ECG systems.
    If the dominant QRS deflection is negative (inverted signal), flip it.
    Detects polarity by comparing the magnitude of the max positive vs max negative peak.
    """
    if np.abs(np.min(signal)) > np.abs(np.max(signal)):
        return -signal
    return signal


def preprocess_signal(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Bandpass + notch filter + polarity normalization + normalize."""
    if len(signal) < 10:
        return signal
    nyq = fs / 2
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    sig = filtfilt(b, a, signal)
    # Notch 60Hz
    b2, a2 = butter(2, [59 / nyq, 61 / nyq], btype='bandstop')
    sig = filtfilt(b2, a2, sig)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    # Polarity normalization — flip if R peaks point downward
    sig = normalize_polarity(sig)
    return sig.astype(np.float32)


def detect_r_peaks(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    R-peak detection using direct peak finding on the filtered signal.
    400ms minimum distance between peaks (max ~150 bpm) prevents T-wave detection.
    Signal must be polarity-normalized before calling this.
    """
    threshold = max(0.3, np.mean(signal) + 0.5 * np.std(signal))

    peaks, _ = find_peaks(
        signal,
        height=threshold,
        distance=int(0.4 * fs),   # 400ms min = max 150 bpm, avoids T-wave
        prominence=0.3,
    )

    # If we got too few peaks, lower the threshold and try again
    if len(peaks) < 3:
        threshold = np.mean(signal) + 0.3 * np.std(signal)
        peaks, _ = find_peaks(
            signal,
            height=threshold,
            distance=int(0.4 * fs),
        )

    return peaks


def compute_rr(peaks: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000
    return rr[(rr > 300) & (rr < 2000)]


def extract_features(rr: np.ndarray) -> dict:
    """HRV features for RF model."""
    if len(rr) < 3:
        return {k: 0.0 for k in _load_meta()["feature_cols"]}

    diff_rr  = np.diff(rr)
    mean_rr  = float(np.mean(rr))
    sdnn     = float(np.std(rr))
    rmssd    = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
    nn50     = int(np.sum(np.abs(diff_rr) > 50))
    pnn50    = float(nn50 / len(diff_rr)) if len(diff_rr) > 0 else 0.0
    cv       = float(sdnn / mean_rr) if mean_rr > 0 else 0.0
    mean_hr  = float(60000 / mean_rr) if mean_rr > 0 else 0.0
    sd1      = float(np.std(diff_rr) / np.sqrt(2))
    sd2      = float(np.sqrt(max(2 * sdnn**2 - sd1**2, 0)))
    sd_ratio = float(sd1 / sd2) if sd2 > 0 else 0.0

    from scipy.stats import entropy as sp_entropy
    hist_counts, _ = np.histogram(rr, bins=min(20, len(rr) // 2))
    hist_counts = hist_counts[hist_counts > 0]
    hist_ent = float(sp_entropy(hist_counts)) if len(hist_counts) > 1 else 0.0

    prop_short = float(np.mean(rr < mean_rr * 0.85))
    prop_long  = float(np.mean(rr > mean_rr * 1.15))

    return {
        "mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd,
        "pnn50": pnn50, "cv": cv, "mean_hr": mean_hr,
        "sd1": sd1, "sd2": sd2, "sd_ratio": sd_ratio,
        "sample_ent": 0.0, "hist_ent": hist_ent,
        "prop_short": prop_short, "prop_long": prop_long,
        "nn50": float(nn50), "n_beats": float(len(rr)),
    }


# ─── Real Demo Signal Loader ──────────────────────────────────────────────────

def load_real_demo(mode: str = "normal") -> np.ndarray:
    """
    Load a real segment from the processed PhysioNet dataset for demo.
    Falls back to None if data not available.
    """
    try:
        signals = np.load(PROCESSED_DIR / "signals.npy")
        labels  = np.load(PROCESSED_DIR / "labels.npy")
        target  = 1 if mode == "afib" else 0
        indices = np.where(labels == target)[0]
        if len(indices) == 0:
            return None
        rng = np.random.default_rng(0 if mode == "normal" else 99)
        idx = rng.choice(indices)
        return signals[idx].astype(np.float32)
    except Exception:
        return None


# ─── Main Predict API ─────────────────────────────────────────────────────────

def predict(signal: np.ndarray, fs: int = SAMPLE_RATE,
            use_model: str = "auto") -> dict:
    """
    Run AFib prediction on a raw ECG signal array.

    Args:
        signal:    1D numpy array of ECG values
        fs:        Sampling rate in Hz (default 250)
        use_model: "rf" | "cnn" | "auto" (prefers CNN if available)

    Returns:
        dict with afib_probability, classification, hrv_features, rr_intervals
    """
    # Preprocess (includes polarity normalization)
    sig = preprocess_signal(signal, fs)

    # Detect R-peaks and compute RR intervals
    peaks = detect_r_peaks(sig, fs)
    rr    = compute_rr(peaks, fs)
    hrv   = extract_features(rr)

    afib_prob = None

    # Try CNN first
    if use_model in ("cnn", "auto"):
        cnn = _load_cnn()
        if cnn is not None:
            try:
                import torch
                if len(sig) < SEGMENT_LEN:
                    sig_in = np.pad(sig, (0, SEGMENT_LEN - len(sig)))
                else:
                    sig_in = sig[:SEGMENT_LEN]
                x = torch.FloatTensor(sig_in[None, None, :])   # (1, 1, L)
                with torch.no_grad():
                    logit = cnn(x)
                    afib_prob = float(torch.sigmoid(logit).item())
            except Exception:
                afib_prob = None

    # Fall back to RF
    if afib_prob is None and use_model in ("rf", "auto"):
        rf = _load_rf()
        if rf is not None:
            meta = _load_meta()
            feat_vec = np.array([hrv.get(f, 0.0) for f in meta["feature_cols"]])
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
            afib_prob = float(rf.predict_proba(feat_vec.reshape(1, -1))[0, 1])

    # Heuristic fallback (no model loaded)
    if afib_prob is None:
        cv        = hrv.get("cv", 0.0)
        rmssd_val = hrv.get("rmssd", 0.0)
        afib_prob = min(0.99, max(0.01, cv * 2.5 + rmssd_val / 500))

    # Classification
    if afib_prob < 0.35:
        classification = "Normal"
        confidence = "High" if afib_prob < 0.15 else "Medium"
    elif afib_prob < 0.65:
        classification = "Borderline"
        confidence = "Low"
    else:
        classification = "AFib"
        confidence = "High" if afib_prob > 0.85 else "Medium"

    hr = hrv.get("mean_hr", 0.0)

    return {
        "afib_probability":  round(afib_prob, 4),
        "classification":    classification,
        "confidence":        confidence,
        "heart_rate":        round(hr, 1),
        "hrv_features":      hrv,
        "rr_intervals":      rr.tolist(),
        "r_peaks":           peaks.tolist(),
        "n_beats":           len(peaks),
        "signal_quality":    _signal_quality(sig, peaks),
        "signal":            sig.tolist(),   # return normalized+flipped signal for display
    }


def _signal_quality(signal: np.ndarray, peaks: np.ndarray) -> str:
    """Rough signal quality estimate."""
    if len(peaks) < 3:
        return "Poor"
    rr = compute_rr(peaks)
    if len(rr) < 2:
        return "Poor"
    noise_ratio = np.std(signal) / (np.max(signal) - np.min(signal) + 1e-8)
    if noise_ratio > 0.3:
        return "Poor"
    if noise_ratio > 0.15:
        return "Fair"
    return "Good"