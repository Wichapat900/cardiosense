"""
preprocess.py — CardioSense Data Preprocessing
================================================
Downloads PhysioNet AF Database and prepares:
  - data/processed/signals.npy    (N, 7500) float32
  - data/processed/labels.npy     (N,) int   0=Normal 1=AFib
  - data/processed/patient_ids.npy (N,) int  for patient-wise splitting

Usage:
  python src/preprocess.py

Known dataset issues handled:
  - Bad blocks 04043, 08405, 08434 — interpolated
  - Record 05091 — uses qrsc annotations
  - Missing signals 00735, 03665 — excluded
"""

import numpy as np
import wfdb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE  = 250
SEG_LEN      = 30        # seconds per segment
SEG_SAMPLES  = SEG_LEN * SAMPLE_RATE   # 7500
OVERLAP      = 0.5       # 50% overlap between segments
DB_PATH      = Path("data/raw/afdb")
OUT_PATH     = Path("data/processed")

# Records known to have issues
EXCLUDE      = {"00735", "03665"}
ALT_ANN      = {"05091": "qrsc"}   # use alternate annotation
BAD_BLOCKS   = {"04043", "08405", "08434"}

# PhysioNet AFDB record list
ALL_RECORDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936",
    "05091", "05121", "05261", "06426", "06453", "06995", "07162",
    "07859", "07879", "07910", "08215", "08219", "08378", "08405",
    "08434", "08455", "08465", "08475",
]
RECORDS = [r for r in ALL_RECORDS if r not in EXCLUDE]

# Rhythm annotation labels
AFIB_LABELS  = {"(AFIB", "AFIB"}
NORMAL_LABELS = {"(N", "N", "(NSR", "NSR"}


def download_record(record_id):
    """Download a single record from PhysioNet."""
    DB_PATH.mkdir(parents=True, exist_ok=True)
    try:
        wfdb.dl_database("afdb", str(DB_PATH), records=[record_id])
        return True
    except Exception as e:
        print(f"  Warning: could not download {record_id}: {e}")
        return False


def load_record(record_id):
    """Load signal and rhythm annotations for one record."""
    rec_path = str(DB_PATH / record_id)

    # Download if not present
    if not (DB_PATH / f"{record_id}.dat").exists():
        print(f"  Downloading {record_id}...")
        if not download_record(record_id):
            return None, None, None

    try:
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal[:, 0].astype(np.float32)

        # Use alternate annotation file if needed
        ann_ext = ALT_ANN.get(record_id, "atr")
        ann = wfdb.rdann(rec_path, ann_ext)

        return signal, ann, record.fs
    except Exception as e:
        print(f"  Error loading {record_id}: {e}")
        return None, None, None


def get_rhythm_map(ann, signal_len):
    """Build a sample-level rhythm label array from annotations."""
    labels = np.full(signal_len, -1, dtype=np.int8)  # -1 = unknown
    current_label = -1

    for i, sym in enumerate(ann.aux_note):
        sym_clean = sym.strip().rstrip("\x00")
        sample = ann.sample[i]

        if sym_clean in AFIB_LABELS:
            current_label = 1
        elif sym_clean in NORMAL_LABELS:
            current_label = 0
        elif sym_clean in ("(AFL", "AFL", "(J", "J"):
            current_label = -1  # exclude AFL and junctional

        if 0 <= sample < signal_len:
            labels[sample] = current_label

    # Forward-fill labels
    cur = -1
    for i in range(signal_len):
        if labels[i] != -1:
            cur = labels[i]
        labels[i] = cur

    return labels


def bandpass_filter(signal, fs, lo=0.5, hi=40.0):
    from scipy.signal import butter, filtfilt
    nyq = fs / 2
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def fix_bad_blocks(signal):
    """Interpolate flat/NaN blocks (common in some AFDB records)."""
    sig = signal.copy()
    # Fix NaN
    nans = np.isnan(sig)
    if nans.any():
        idx = np.arange(len(sig))
        sig[nans] = np.interp(idx[nans], idx[~nans], sig[~nans])
    # Fix flat blocks (>500 identical consecutive samples)
    i = 0
    while i < len(sig) - 1:
        if sig[i] == sig[i + 1]:
            j = i + 1
            while j < len(sig) and sig[j] == sig[i]:
                j += 1
            if j - i > 500:
                # Interpolate over the flat region
                if i > 0 and j < len(sig):
                    sig[i:j] = np.linspace(sig[i - 1], sig[j], j - i)
            i = j
        else:
            i += 1
    return sig


def extract_segments(signal, rhythm_map, fs, record_idx):
    """Slice signal into overlapping segments with majority-vote labels."""
    step = int(SEG_SAMPLES * (1 - OVERLAP))
    segments, labels, pids = [], [], []

    for start in range(0, len(signal) - SEG_SAMPLES, step):
        end = start + SEG_SAMPLES
        seg_labels = rhythm_map[start:end]

        # Skip segments with unknown labels
        known = seg_labels[seg_labels != -1]
        if len(known) < SEG_SAMPLES * 0.8:
            continue

        # Majority vote — must be >80% one class
        afib_frac = np.mean(known == 1)
        if afib_frac > 0.8:
            lbl = 1
        elif afib_frac < 0.2:
            lbl = 0
        else:
            continue  # ambiguous segment — skip

        seg = signal[start:end].copy()

        # Resample if needed
        if fs != SAMPLE_RATE:
            from scipy.signal import resample
            seg = resample(seg, SEG_SAMPLES).astype(np.float32)

        # Filter
        seg = bandpass_filter(seg, SAMPLE_RATE).astype(np.float32)

        # Normalise
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)

        segments.append(seg)
        labels.append(lbl)
        pids.append(record_idx)

    return segments, labels, pids


def preprocess():
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    all_segs, all_labels, all_pids = [], [], []

    for rec_idx, record_id in enumerate(RECORDS):
        print(f"[{rec_idx+1}/{len(RECORDS)}] Processing {record_id}...")

        signal, ann, fs = load_record(record_id)
        if signal is None:
            continue

        # Fix bad data blocks
        if record_id in BAD_BLOCKS:
            signal = fix_bad_blocks(signal)

        rhythm_map = get_rhythm_map(ann, len(signal))
        segs, lbls, pids = extract_segments(signal, rhythm_map, fs, rec_idx)

        print(f"  → {len(segs)} segments  "
              f"AFib={sum(l==1 for l in lbls)}  "
              f"Normal={sum(l==0 for l in lbls)}")

        all_segs.extend(segs)
        all_labels.extend(lbls)
        all_pids.extend(pids)

    X = np.array(all_segs,   dtype=np.float32)
    y = np.array(all_labels, dtype=np.int8)
    g = np.array(all_pids,   dtype=np.int16)

    np.save(OUT_PATH / "signals.npy",     X)
    np.save(OUT_PATH / "labels.npy",      y)
    np.save(OUT_PATH / "patient_ids.npy", g)

    print(f"\n{'='*50}")
    print(f"Saved {len(X)} segments")
    print(f"  AFib:   {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
    print(f"  Normal: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
    print(f"  Shape:  {X.shape}")
    print(f"Saved to {OUT_PATH}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    preprocess()