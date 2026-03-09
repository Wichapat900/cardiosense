"""
predict.py — CardioSense Inference + SHAP Explainability
=========================================================
Runs the explainable Random Forest pipeline and returns:
  - AFib probability + classification
  - Full HRV feature values
  - SHAP values (feature contributions to the decision)
  - Signal quality metrics
  - R-peaks for display
"""

import numpy as np
import joblib
import shap
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt

# Import feature extraction from train.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import extract_features, detect_rpeaks, FEATURE_NAMES

SAMPLE_RATE = 250
MODEL_PATH  = Path("models/saved/rf_pipeline.pkl")

_bundle  = None
_explainer = None


def _load():
    global _bundle, _explainer
    if _bundle is None:
        _bundle   = joblib.load(MODEL_PATH)
        # Build SHAP TreeExplainer once — fast for Random Forest
        _explainer = shap.TreeExplainer(_bundle["model"])
    return _bundle, _explainer


def signal_quality(signal, fs=SAMPLE_RATE):
    """Estimate signal quality: Good / Fair / Poor."""
    sig = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    noise = np.std(np.diff(sig))
    if noise < 0.3:
        return "Good"
    elif noise < 0.6:
        return "Fair"
    return "Poor"


def predict(signal: np.ndarray, fs: int = SAMPLE_RATE) -> dict:
    """
    Run full inference pipeline.

    Returns
    -------
    dict with keys:
      afib_probability  : float 0-1
      classification    : "Normal" | "Borderline" | "AFib"
      confidence        : "High" | "Medium" | "Low"
      heart_rate        : float bpm
      hrv_features      : dict of 20 features
      shap_values       : dict {feature_name: shap_value}
      shap_base_value   : float (model baseline probability)
      r_peaks           : list of sample indices
      rr_intervals      : list of RR intervals in ms
      n_beats           : int
      signal_quality    : str
      signal            : list (normalised signal for display)
    """
    bundle, explainer = _load()
    rf      = bundle["model"]
    scaler  = bundle["scaler"]
    thresh  = bundle["threshold"]

    # ── Feature extraction ────────────────────────────────────────────────────
    feats = extract_features(signal, fs)
    if feats is None:
        return _fallback(signal, fs)

    feat_vec = np.array([[feats[f] for f in FEATURE_NAMES]], dtype=np.float32)
    feat_scaled = scaler.transform(feat_vec)

    # ── Prediction ────────────────────────────────────────────────────────────
    prob = float(rf.predict_proba(feat_scaled)[0, 1])

    if prob >= thresh:
        cls = "AFib"
    elif prob >= thresh * 0.6:
        cls = "Borderline"
    else:
        cls = "Normal"

    # Confidence based on distance from threshold
    dist = abs(prob - thresh)
    confidence = "High" if dist > 0.25 else ("Medium" if dist > 0.10 else "Low")

    # ── SHAP values ───────────────────────────────────────────────────────────
    shap_vals = explainer.shap_values(feat_scaled)
    # shap_values returns [class0_shap, class1_shap] for RF
    if isinstance(shap_vals, list):
        shap_afib = shap_vals[1][0]   # class 1 (AFib) SHAP values
    else:
        shap_afib = shap_vals[0]

    shap_dict = {name: float(val)
                 for name, val in zip(FEATURE_NAMES, shap_afib)}

    base_val = float(explainer.expected_value[1]
                     if isinstance(explainer.expected_value, (list, np.ndarray))
                     else explainer.expected_value)

    # ── R-peaks and RR intervals ──────────────────────────────────────────────
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
    """Fallback when too few beats detected."""
    peaks, sig_norm = detect_rpeaks(signal, fs)
    return {
        "afib_probability": 0.0,
        "classification":   "Insufficient Data",
        "confidence":       "Low",
        "heart_rate":       0.0,
        "hrv_features":     {f: 0.0 for f in FEATURE_NAMES},
        "shap_values":      {f: 0.0 for f in FEATURE_NAMES},
        "shap_base_value":  0.5,
        "r_peaks":          peaks.tolist(),
        "rr_intervals":     [],
        "n_beats":          len(peaks),
        "signal_quality":   "Poor",
        "signal":           sig_norm.tolist(),
    }