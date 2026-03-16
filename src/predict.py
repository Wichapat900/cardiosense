"""
predict.py — CardioSense Inference + SHAP Explainability
=========================================================
"""

import numpy as np
import joblib
import shap
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import extract_features, detect_rpeaks, FEATURE_NAMES

SAMPLE_RATE = 250
MODEL_PATH  = Path("models/saved/rf_pipeline.pkl")

_bundle    = None
_explainer = None


def _load():
    global _bundle, _explainer
    if _bundle is None:
        _bundle    = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_bundle["model"])
    return _bundle, _explainer


def signal_quality(signal, fs=SAMPLE_RATE):
    sig = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    noise = np.std(np.diff(sig))
    if noise < 0.3:
        return "Good"
    elif noise < 0.6:
        return "Fair"
    return "Poor"


def predict(signal: np.ndarray, fs: int = SAMPLE_RATE) -> dict:
    bundle, explainer = _load()
    rf     = bundle["model"]
    scaler = bundle["scaler"]
    thresh = bundle["threshold"]

    feats = extract_features(signal, fs)
    if feats is None:
        return _fallback(signal, fs)

    feat_vec    = np.array([[feats[f] for f in FEATURE_NAMES]], dtype=np.float32)
    feat_scaled = scaler.transform(feat_vec)

    # Prediction
    prob = float(rf.predict_proba(feat_scaled)[0, 1])
    if prob >= thresh:
        cls = "AFib"
    elif prob >= thresh * 0.6:
        cls = "Borderline"
    else:
        cls = "Normal"

    dist = abs(prob - thresh)
    confidence = "High" if dist > 0.25 else ("Medium" if dist > 0.10 else "Low")

    # SHAP — always extract class 1 (AFib) regardless of shap version
    shap_vals = explainer.shap_values(feat_scaled)
    if isinstance(shap_vals, list):
        # Old shap: [class0_array, class1_array]
        shap_afib = np.array(shap_vals[1]).flatten()
    else:
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            # New shap: (n_samples, n_features, n_classes)
            shap_afib = arr[0, :, 1]
        elif arr.ndim == 2:
            # (n_classes, n_features)
            shap_afib = arr[1].flatten()
        else:
            shap_afib = arr.flatten()

    shap_dict = {name: float(np.array(val).flatten()[0])
                 for name, val in zip(FEATURE_NAMES, shap_afib)}

    # Base value — class 1 (AFib) expected value
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev_arr = np.array(ev).flatten()
        base_val = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
    else:
        base_val = float(ev)

    peaks, sig_norm = detect_rpeaks(signal, fs)
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 300) & (rr < 2000)]
    hr = float(60000 / np.mean(rr)) if len(rr) > 0 else 0.0

    return {
        "afib_probability": round(prob, 4),
        "classification":   cls,
        "confidence":       confidence,
        "heart_rate":       round(hr, 1),
        "hrv_features":     feats,
        "shap_values":      shap_dict,
        "shap_base_value":  base_val,
        "r_peaks":          peaks.tolist(),
        "rr_intervals":     rr.tolist(),
        "n_beats":          len(peaks),
        "signal_quality":   signal_quality(signal, fs),
        "signal":           sig_norm.tolist(),
    }


def _fallback(signal, fs):
    peaks, sig_norm = detect_rpeaks(signal, fs)
    return {
        "afib_probability": 0.0,
        "classification":   "Insufficient Data",
        "confidence":       "Low",
        "heart_rate":       0.0,
        "hrv_features":     {f: 0.0 for f in FEATURE_NAMES},
        "shap_values":      {f: 0.0 for f in FEATURE_NAMES},
        "shap_base_value":  0.3,
        "r_peaks":          peaks.tolist(),
        "rr_intervals":     [],
        "n_beats":          len(peaks),
        "signal_quality":   "Poor",
        "signal":           sig_norm.tolist(),
    }