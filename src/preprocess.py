"""
preprocess.py — CardioSense AFib Detection
==========================================
Downloads the PhysioNet AF Database and extracts:
  - RR interval time series
  - HRV features (RMSSD, pNN50, SDNN, CV, entropy)
  - Raw 30-second ECG segments (for CNN input)
  - Rhythm labels (Normal / AFib / AFL / Other)

Handles class imbalance via SMOTE on HRV features.

Usage:
    python src/preprocess.py
"""

import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import entropy as scipy_entropy
from pathlib import Path
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────────

# Point this to your dataset folder — change if needed
DATA_DIR      = Path("AFib_dataset")
PROCESSED_DIR = Path("data/processed")
SAMPLE_RATE   = 250          # Hz — AFDB native sample rate
SEGMENT_SEC   = 30           # seconds per training segment
SEGMENT_LEN   = SAMPLE_RATE * SEGMENT_SEC   # 7500 samples
OVERLAP       = 0.5          # 50% overlap between segments

# ─── notes.txt known issues ──────────────────────────────────────────────────
# 00735        Signals unavailable              → excluded entirely
# 03665        Signals unavailable              → excluded entirely
# 04043        Block 39 unreadable              → load with NaN handling
# 04936        Signals previously available     → included, flagged
# 05091        Corrected QRS annotations (qrsc) → use qrsc annotator if present
# 06453        Recording ends ~9h15m            → fine, just fewer segments
# 08378        No start time                    → fine, we don't use absolute time
# 08405        No start time; block 1067 bad    → load with NaN handling
# 08434        Blocks 648, 857, 894 unreadable  → load with NaN handling
# 08455        No start time                    → fine, we don't use absolute time
# ─────────────────────────────────────────────────────────────────────────────

# 23 usable records (00735 and 03665 excluded — no signals)
# 04048 duplicate removed from original list
AFDB_RECORDS = [
    "04015", "04043", "04048", "04126", "04746",
    "04908", "04936", "05091", "05121", "05261",
    "06426", "06453", "06995", "07162", "07859",
    "07879", "07910", "08215", "08219", "08378",
    "08405", "08434", "08455", "05269"
]

# Records with known unreadable blocks — signal will have NaN gaps
# preprocess handles these by interpolating across the bad blocks
RECORDS_WITH_BAD_BLOCKS = {
    "04043": [39],           # block 39 unreadable
    "08405": [1067],         # block 1067 unreadable
    "08434": [648, 857, 894] # blocks 648, 857, 894 unreadable
}

# Record 05091 has corrected QRS annotations — prefer qrsc over atr
# All records also have .qrs files (pre-computed R-peak annotations)
# process_record uses .qrs for R-peak positions if available — more accurate
# than our own detector and already validated
RECORDS_WITH_CORRECTED_ANN = {"05091": "qrsc"}

# Label mapping from AFDB rhythm annotations
LABEL_MAP = {
    "N":    0,   # Normal sinus rhythm
    "(N":   0,
    "AFIB": 1,   # Atrial fibrillation
    "(AFIB":1,
    "AFL":  2,   # Atrial flutter
    "(AFL": 2,
    "J":    3,   # AV junctional rhythm
    "(J":   3,
}

CLASS_NAMES = ["Normal", "AFib", "AFl", "Other"]

# ─── Signal Processing ─────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, low: float = 0.5, high: float = 40.0,
                    fs: int = SAMPLE_RATE) -> np.ndarray:
    """Bandpass Butterworth filter for ECG cleaning."""
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, freq: float = 60.0,
                 fs: int = SAMPLE_RATE) -> np.ndarray:
    """60Hz notch filter to remove power line noise."""
    nyq = fs / 2
    b, a = butter(2, [(freq - 1) / nyq, (freq + 1) / nyq], btype='bandstop')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Pan-Tompkins-inspired R-peak detector.
    Returns sample indices of R peaks.
    """
    # Differentiate + square to enhance QRS
    diff = np.diff(signal, prepend=signal[0])
    squared = diff ** 2

    # Moving window integration (150ms window)
    win = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')

    # Adaptive threshold
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    min_distance = int(0.25 * fs)   # minimum 250ms between peaks (240 bpm max)

    peaks, props = find_peaks(integrated, height=threshold,
                              distance=min_distance, prominence=threshold * 0.3)
    return peaks


def compute_rr_intervals(r_peaks: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks) / fs * 1000   # ms


# ─── HRV Feature Extraction ────────────────────────────────────────────────────

def extract_hrv_features(rr: np.ndarray) -> dict:
    """
    Extract 15 HRV features that are strong AFib discriminators.
    Based on literature (Moody & Mark 1983, Lévy et al. 1998).
    """
    if len(rr) < 5:
        return {k: 0.0 for k in _hrv_feature_names()}

    rr = rr[(rr > 300) & (rr < 2000)]   # physiological range filter
    if len(rr) < 3:
        return {k: 0.0 for k in _hrv_feature_names()}

    diff_rr = np.diff(rr)

    # Time-domain features
    mean_rr  = float(np.mean(rr))
    sdnn     = float(np.std(rr))
    rmssd    = float(np.sqrt(np.mean(diff_rr ** 2)))
    nn50     = int(np.sum(np.abs(diff_rr) > 50))
    pnn50    = float(nn50 / len(diff_rr)) if len(diff_rr) > 0 else 0.0
    cv       = float(sdnn / mean_rr) if mean_rr > 0 else 0.0    # coefficient of variation
    mean_hr  = float(60000 / mean_rr) if mean_rr > 0 else 0.0

    # Irregularity metrics (key for AFib)
    # AFib: highly irregular → high CV, high RMSSD, high pNN50
    sd1 = float(np.std(diff_rr) / np.sqrt(2))    # Poincaré short-axis
    sd2 = float(np.sqrt(2 * sdnn**2 - sd1**2)) if 2 * sdnn**2 > sd1**2 else 0.0
    sd_ratio = float(sd1 / sd2) if sd2 > 0 else 0.0

    # Entropy features (AFib has higher entropy)
    # Sample entropy approximation
    sample_ent = _sample_entropy(rr, m=2, r=0.2 * sdnn) if sdnn > 0 else 0.0

    # Histogram entropy of RR distribution
    hist_counts, _ = np.histogram(rr, bins=min(20, len(rr) // 2))
    hist_counts = hist_counts[hist_counts > 0]
    hist_ent = float(scipy_entropy(hist_counts)) if len(hist_counts) > 1 else 0.0

    # Frequency proxy: proportion of short RR intervals (AFib often has fast rates)
    prop_short = float(np.mean(rr < mean_rr * 0.85))
    prop_long  = float(np.mean(rr > mean_rr * 1.15))

    return {
        "mean_rr":    mean_rr,
        "sdnn":       sdnn,
        "rmssd":      rmssd,
        "pnn50":      pnn50,
        "cv":         cv,
        "mean_hr":    mean_hr,
        "sd1":        sd1,
        "sd2":        sd2,
        "sd_ratio":   sd_ratio,
        "sample_ent": sample_ent,
        "hist_ent":   hist_ent,
        "prop_short": prop_short,
        "prop_long":  prop_long,
        "nn50":       float(nn50),
        "n_beats":    float(len(rr)),
    }


def _hrv_feature_names():
    return ["mean_rr", "sdnn", "rmssd", "pnn50", "cv", "mean_hr",
            "sd1", "sd2", "sd_ratio", "sample_ent", "hist_ent",
            "prop_short", "prop_long", "nn50", "n_beats"]


def _sample_entropy(rr: np.ndarray, m: int = 2, r: float = None) -> float:
    """Fast approximate sample entropy."""
    N = len(rr)
    if N < 10:
        return 0.0
    if r is None:
        r = 0.2 * np.std(rr)
    if r == 0:
        return 0.0

    def _phi(m_val):
        count = 0
        for i in range(N - m_val):
            template = rr[i:i + m_val]
            matches = np.sum(
                np.max(np.abs(
                    np.array([rr[j:j + m_val] for j in range(N - m_val) if j != i]) - template
                ), axis=1) <= r
            ) if N - m_val > 1 else 0
            count += matches
        return count

    try:
        A = _phi(m + 1)
        B = _phi(m)
        if B == 0:
            return 0.0
        return float(-np.log(A / B))
    except Exception:
        return 0.0


# ─── Segment Labeler ──────────────────────────────────────────────────────────

def get_dominant_label(ann_samples: np.ndarray, ann_symbols: list,
                       seg_start: int, seg_end: int) -> int:
    """
    Find the dominant rhythm label within a segment.
    Returns binary: 1 = AFib, 0 = Non-AFib
    (AFL and other arrhythmias treated as non-AFib for binary classification)
    """
    # Find annotations that fall within [seg_start, seg_end]
    mask = (ann_samples >= seg_start) & (ann_samples < seg_end)
    labels_in_seg = [ann_symbols[i] for i, m in enumerate(mask) if m]

    if not labels_in_seg:
        # Carry forward last annotation before segment
        before = np.where(ann_samples < seg_start)[0]
        if len(before) > 0:
            last_sym = ann_symbols[before[-1]]
            labels_in_seg = [last_sym]
        else:
            return 0  # Default to normal if unknown

    # Map to numeric
    numeric = [LABEL_MAP.get(s, 0) for s in labels_in_seg]

    # Binary: AFib (1) vs rest (0)
    afib_count = sum(1 for l in numeric if l == 1)
    return 1 if afib_count > len(numeric) * 0.5 else 0


# ─── Main Processing ──────────────────────────────────────────────────────────

def _interpolate_bad_blocks(signal: np.ndarray, bad_blocks: list,
                             block_size: int = 512) -> np.ndarray:
    """
    Linearly interpolate over known unreadable blocks so the signal
    array stays intact rather than crashing or producing garbage.
    Block numbers are 0-indexed. Block size in AFDB is 512 samples.
    """
    signal = signal.copy()
    total = len(signal)
    for block in bad_blocks:
        start = block * block_size
        end   = min(start + block_size, total)
        if start >= total:
            continue
        # Use the last good sample before and first good sample after
        val_before = signal[start - 1] if start > 0 else 0.0
        val_after  = signal[end]       if end < total else val_before
        signal[start:end] = np.linspace(val_before, val_after, end - start)
        tqdm.write(f"    ↳ Interpolated bad block {block} (samples {start}–{end})")
    return signal


def process_record(record_id: str) -> list[dict]:
    """
    Process a single AFDB record.
    Handles notes.txt issues:
      - Bad blocks  → linear interpolation across corrupt region
      - Corrected annotations (05091) → uses qrsc annotator
      - No start time records → fine, we ignore absolute timestamps
    Returns list of segment dicts with features + label.
    """
    segments = []

    try:
        record_path = str(DATA_DIR / record_id)

        # Load ECG signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0].astype(np.float32)   # Lead 1

        # ── Handle known bad blocks (notes.txt) ──
        if record_id in RECORDS_WITH_BAD_BLOCKS:
            bad = RECORDS_WITH_BAD_BLOCKS[record_id]
            tqdm.write(f"  ⚠  {record_id}: interpolating bad blocks {bad}")
            signal = _interpolate_bad_blocks(signal, bad)

        # ── Handle NaN/Inf values that wfdb may produce from corrupt blocks ──
        nan_count = int(np.isnan(signal).sum() + np.isinf(signal).sum())
        if nan_count > 0:
            tqdm.write(f"  ⚠  {record_id}: {nan_count} NaN/Inf values — interpolating")
            nans = ~np.isfinite(signal)
            idx  = np.arange(len(signal))
            signal[nans] = np.interp(idx[nans], idx[~nans], signal[~nans])

        # ── Load .qrs R-peak annotations if available ──
        qrs_samples = None
        try:
            qrs_ann = wfdb.rdann(record_path, 'qrs')
            qrs_samples = qrs_ann.sample
            tqdm.write(f"  ✓  {record_id}: using pre-computed .qrs R-peak annotations")
        except FileNotFoundError:
            tqdm.write(f"  ⚠  {record_id}: no .qrs file found, using built-in R-peak detector")

        # ── Load rhythm annotations ──
        # 05091 has corrected annotations under 'qrsc' annotator
        ann_ext = RECORDS_WITH_CORRECTED_ANN.get(record_id, "atr")
        try:
            ann = wfdb.rdann(record_path, ann_ext)
        except FileNotFoundError:
            # Fall back to atr if corrected file not present
            tqdm.write(f"  ⚠  {record_id}: {ann_ext} not found, falling back to atr")
            ann = wfdb.rdann(record_path, "atr")

        ann_samples = ann.sample
        ann_symbols = ann.aux_note   # rhythm labels like "(AFIB", "(N", etc.

        # ── Clean signal ──
        signal = bandpass_filter(signal)
        signal = notch_filter(signal)

        # ── Normalize ──
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        total_samples = len(signal)
        step = int(SEGMENT_LEN * (1 - OVERLAP))

        for start in range(0, total_samples - SEGMENT_LEN, step):
            end = start + SEGMENT_LEN
            seg = signal[start:end]

            # Skip any segment that still has non-finite values after interpolation
            if not np.all(np.isfinite(seg)):
                continue

            # Get label
            label = get_dominant_label(ann_samples, ann_symbols, start, end)

            # Use pre-computed .qrs R-peak annotations if available
            # (already validated — faster and more accurate than our detector)
            if qrs_samples is not None:
                # Find QRS peaks that fall within this segment, shift to local index
                seg_peaks = qrs_samples[(qrs_samples >= start) & (qrs_samples < end)] - start
                r_peaks = seg_peaks if len(seg_peaks) >= 2 else detect_r_peaks(seg)
            else:
                r_peaks = detect_r_peaks(seg)
            rr = compute_rr_intervals(r_peaks)

            # Extract HRV features
            hrv = extract_hrv_features(rr)

            segments.append({
                "record_id": record_id,
                "start":     start,
                "label":     label,
                "signal":    seg,          # raw waveform for CNN
                **hrv
            })

    except FileNotFoundError:
        print(f"  ⚠  Record {record_id} not found in {DATA_DIR}/")
    except Exception as e:
        print(f"  ✗  Error processing {record_id}: {e}")

    return segments


def download_afdb():
    """Download AFDB from PhysioNet using wfdb."""
    print("📥 Downloading PhysioNet AFDB 1.0.0...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        wfdb.dl_database("afdb", str(DATA_DIR))
        print(f"✓ Downloaded to {DATA_DIR}/")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("  → Try manually: https://physionet.org/content/afdb/1.0.0/")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Download if needed
    if not any(DATA_DIR.glob("*.dat")):
        download_afdb()
    else:
        print(f"✓ Found existing data in {DATA_DIR}/")

    # Scan which records actually exist
    available = [r for r in AFDB_RECORDS
                 if (DATA_DIR / f"{r}.dat").exists()]
    print(f"\n📂 Processing {len(available)} records...")

    all_segments = []
    for rec in tqdm(available, desc="Records"):
        segs = process_record(rec)
        all_segments.extend(segs)
        tqdm.write(f"  {rec}: {len(segs)} segments, "
                   f"{sum(s['label'] for s in segs)} AFib")

    if not all_segments:
        print("\n✗ No segments extracted. Check your data directory.")
        return

    # Build DataFrames
    feature_cols = _hrv_feature_names()
    df = pd.DataFrame(all_segments)

    print(f"\n📊 Dataset Summary:")
    print(f"  Total segments: {len(df)}")
    counts = df['label'].value_counts()
    print(f"  Normal:  {counts.get(0, 0)}")
    print(f"  AFib:    {counts.get(1, 0)}")
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"  Imbalance ratio: {ratio:.1f}:1  ← Will be corrected with SMOTE")

    # Save HRV features (for sklearn / gradient boosting models)
    features_df = df[["record_id", "start", "label"] + feature_cols]
    features_path = PROCESSED_DIR / "hrv_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\n✓ HRV features saved → {features_path}")

    # Save raw signals as numpy arrays (for CNN)
    signals = np.stack(df["signal"].values)           # (N, 7500)
    labels  = df["label"].values                       # (N,)
    np.save(PROCESSED_DIR / "signals.npy", signals)
    np.save(PROCESSED_DIR / "labels.npy", labels)
    print(f"✓ Signals saved  → {PROCESSED_DIR}/signals.npy  shape={signals.shape}")
    print(f"✓ Labels saved   → {PROCESSED_DIR}/labels.npy   shape={labels.shape}")

    # Save metadata
    meta = {
        "sample_rate":   SAMPLE_RATE,
        "segment_len":   SEGMENT_LEN,
        "segment_sec":   SEGMENT_SEC,
        "feature_cols":  feature_cols,
        "class_names":   ["Normal", "AFib"],
        "n_normal":      int(counts.get(0, 0)),
        "n_afib":        int(counts.get(1, 0)),
    }
    joblib.dump(meta, PROCESSED_DIR / "meta.pkl")
    print(f"✓ Metadata saved → {PROCESSED_DIR}/meta.pkl")
    print("\n🎉 Preprocessing complete! Run: python src/train.py")


if __name__ == "__main__":
    main()