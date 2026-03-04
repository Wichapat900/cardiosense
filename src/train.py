"""
train.py — CardioSense AFib Detection
======================================
Trains a 1D CNN + BiLSTM hybrid model on the PhysioNet AFDB dataset.

Imbalance strategy (three-pronged):
  1. SMOTE oversampling on HRV feature space
  2. Class-weighted loss function
  3. Focal Loss (γ=2) to focus on hard AFib examples

Also trains a lightweight Random Forest on HRV features as a fallback model
(works without GPU, good for Streamlit demo).

Usage:
    python src/train.py
    python src/train.py --model rf          # Random Forest only (fast)
    python src/train.py --model cnn         # CNN+BiLSTM only
    python src/train.py --model both        # Train both (default)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


# ─── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
RESULTS_DIR   = Path("models/results")

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1

# CNN training
CNN_EPOCHS    = 40
CNN_LR        = 1e-3
CNN_BATCH     = 64
CNN_DROPOUT   = 0.4

# SMOTE
SMOTE_RATIO   = 0.8       # AFib will be 80% of Normal count

# Focal Loss
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.75      # Weight for minority class


# ─── Patient-Wise Split ───────────────────────────────────────────────────────

def patient_wise_split(df: pd.DataFrame,
                       test_size: float = 0.20,
                       val_size:  float = 0.10,
                       random_state: int = RANDOM_STATE):
    """
    Split by PATIENT, not by segment.

    All segments from a given record_id go entirely into train, val, or test.
    This prevents data leakage where the model memorises a patient's unique
    ECG signature rather than learning to detect AFib.

    With only 25 AFDB patients we keep it deterministic rather than random
    so results are reproducible and you can inspect exactly which patients
    are in each fold.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of integer positions into df
    patient_split_info           : dict  (for logging / sanity checking)
    """
    rng = np.random.default_rng(random_state)

    # Get unique patients and their dominant label (AFib if >50% of segments are AFib)
    patients = df.groupby("record_id")["label"].agg(
        lambda x: int(x.mean() > 0.5)   # 1 = predominantly AFib patient
    ).reset_index()
    patients.columns = ["record_id", "afib_dominant"]

    afib_patients   = patients[patients["afib_dominant"] == 1]["record_id"].tolist()
    normal_patients = patients[patients["afib_dominant"] == 0]["record_id"].tolist()

    print(f"\n   Patient-wise split info:")
    print(f"     Total patients     : {len(patients)}")
    print(f"     AFib-dominant      : {len(afib_patients)}  → {afib_patients}")
    print(f"     Normal-dominant    : {len(normal_patients)}")

    def split_group(group: list, test_frac: float, val_frac: float):
        rng.shuffle(group := list(group))
        n = len(group)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))
        n_train = n - n_test - n_val
        if n_train < 1:
            # With very few patients put at least 1 in train
            n_test = max(1, n_test - 1)
            n_train = n - n_test - n_val
        return (group[:n_train],
                group[n_train:n_train + n_val],
                group[n_train + n_val:])

    # Split each group separately to preserve class balance across splits
    afib_tr, afib_va, afib_te   = split_group(afib_patients,   test_size, val_size)
    norm_tr, norm_va, norm_te   = split_group(normal_patients,  test_size, val_size)

    train_patients = afib_tr + norm_tr
    val_patients   = afib_va + norm_va
    test_patients  = afib_te + norm_te

    print(f"     Train patients ({len(train_patients)}): {sorted(train_patients)}")
    print(f"     Val   patients ({len(val_patients)}): {sorted(val_patients)}")
    print(f"     Test  patients ({len(test_patients)}): {sorted(test_patients)}")

    # Map back to row indices
    train_idx = df[df["record_id"].isin(train_patients)].index.to_numpy()
    val_idx   = df[df["record_id"].isin(val_patients)].index.to_numpy()
    test_idx  = df[df["record_id"].isin(test_patients)].index.to_numpy()

    # Sanity: no patient appears in more than one split
    assert set(train_patients).isdisjoint(test_patients),  "LEAK: patient in train+test!"
    assert set(train_patients).isdisjoint(val_patients),   "LEAK: patient in train+val!"
    assert set(val_patients).isdisjoint(test_patients),    "LEAK: patient in val+test!"

    info = {
        "train_patients": sorted(train_patients),
        "val_patients":   sorted(val_patients),
        "test_patients":  sorted(test_patients),
        "train_segments": int(len(train_idx)),
        "val_segments":   int(len(val_idx)),
        "test_segments":  int(len(test_idx)),
    }
    return train_idx, val_idx, test_idx, info


def patient_wise_split_signals(df_with_ids: pd.DataFrame,
                                signals: np.ndarray,
                                labels: np.ndarray,
                                test_size: float = 0.20,
                                val_size:  float = 0.10):
    """
    Patient-wise split for raw signal arrays (CNN input).
    df_with_ids must have a 'record_id' column aligned with signals/labels.
    """
    train_idx, val_idx, test_idx, info = patient_wise_split(
        df_with_ids, test_size, val_size
    )
    return (signals[train_idx], signals[val_idx],  signals[test_idx],
            labels[train_idx],  labels[val_idx],   labels[test_idx],
            info)

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch not found — CNN training skipped. Run: pip install torch")


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces loss for well-classified easy examples.
    Focuses training on hard AFib cases.

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


# ─── Model Architecture ───────────────────────────────────────────────────────

class AFibCNNBiLSTM(nn.Module):
    """
    1D CNN feature extractor + BiLSTM temporal model for ECG AFib detection.

    Architecture:
      Conv1D blocks (multi-scale kernel sizes) → BiLSTM → Classifier head

    Input:  (batch, 1, 7500)   ← 30s @ 250Hz
    Output: (batch, 1)         ← AFib logit
    """
    def __init__(self, input_len: int = 7500, dropout: float = CNN_DROPOUT):
        super().__init__()

        # Multi-scale CNN blocks (captures different frequency components)
        self.conv_short = self._conv_block(1, 32, kernel_size=7)    # ~28ms
        self.conv_mid   = self._conv_block(1, 32, kernel_size=25)   # 100ms
        self.conv_long  = self._conv_block(1, 32, kernel_size=75)   # 300ms (QRS width)

        # Merge + reduce
        self.merge_conv = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
        )

        # Deeper feature extraction
        self.deep_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        # BiLSTM for temporal pattern (P-wave absence, irregular RR)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Attention over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _conv_block(self, in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        s = self.conv_short(x)
        m = self.conv_mid(x)
        l = self.conv_long(x)

        # Align lengths (max pool may differ by 1 sample)
        min_len = min(s.size(-1), m.size(-1), l.size(-1))
        merged = torch.cat([s[..., :min_len], m[..., :min_len], l[..., :min_len]], dim=1)

        feat = self.merge_conv(merged)
        feat = self.deep_conv(feat)             # (B, 128, T)

        # LSTM: (B, T, 128)
        lstm_out, _ = self.bilstm(feat.permute(0, 2, 1))

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)   # (B, 128)

        return self.classifier(context).squeeze(-1)      # (B,)


# ─── CNN Training ─────────────────────────────────────────────────────────────

def train_cnn(signals: np.ndarray, labels: np.ndarray, meta: dict) -> dict:
    if not TORCH_AVAILABLE:
        print("✗ PyTorch not available — skip CNN")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Training CNN+BiLSTM on {device}")
    print(f"   Input shape: {signals.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Patient-wise split (prevents data leakage) ──
    # Load the segment metadata (record_id per segment) saved by preprocess.py
    meta_df_path = PROCESSED_DIR / "hrv_features.csv"
    if not meta_df_path.exists():
        print("   ✗ hrv_features.csv not found — cannot do patient-wise split for CNN.")
        return {}

    segment_ids = pd.read_csv(meta_df_path)[["record_id", "label"]]
    assert len(segment_ids) == len(signals), \
        f"Mismatch: {len(segment_ids)} feature rows vs {len(signals)} signal rows"

    X_train, X_val, X_test, y_train, y_val, y_test, split_info = \
        patient_wise_split_signals(segment_ids, signals, labels, TEST_SIZE, VAL_SIZE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "patient_split_cnn.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"   Train: {len(y_train)} segments ({int(y_train.sum())} AFib) "
          f"— {len(split_info['train_patients'])} patients")
    print(f"   Val:   {len(y_val)} segments ({int(y_val.sum())} AFib) "
          f"— {len(split_info['val_patients'])} patients")
    print(f"   Test:  {len(y_test)} segments ({int(y_test.sum())} AFib) "
          f"— {len(split_info['test_patients'])} patients  ← HELD-OUT PATIENTS ONLY")

    # Tensors — shape: (N, 1, L)
    def to_tensor(X, y):
        return (torch.FloatTensor(X[:, None, :]),
                torch.FloatTensor(y))

    Xt, yt = to_tensor(X_train, y_train)
    Xv, yv = to_tensor(X_val, y_val)

    # Weighted sampler to oversample AFib during training
    class_counts = np.bincount(y_train.astype(int))
    sample_weights = [1.0 / class_counts[int(y)] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=CNN_BATCH,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        TensorDataset(Xv, yv),
        batch_size=CNN_BATCH * 2,
        shuffle=False,
    )

    # Model
    model = AFibCNNBiLSTM(input_len=signals.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")

    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(CNN_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)

        # Batch-level progress bar so you can see activity within each epoch
        batch_bar = tqdm(train_loader,
                         desc=f"   Epoch {epoch+1:02d}/{CNN_EPOCHS} [train]",
                         leave=False,
                         ncols=80,
                         unit="batch")

        for xb, yb in batch_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            # Show running loss in the progress bar
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= n_batches
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_preds, val_probs = 0.0, [], []
        with torch.no_grad():
            val_bar = tqdm(val_loader,
                           desc=f"   Epoch {epoch+1:02d}/{CNN_EPOCHS} [val]  ",
                           leave=False,
                           ncols=80,
                           unit="batch")
            for xb, yb in val_bar:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs > 0.5).astype(int))
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(y_val, val_probs)
        val_f1  = f1_score(y_val, val_preds, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        # Clear the tqdm lines then print the clean epoch summary
        print(f"   Epoch {epoch+1:02d}/{CNN_EPOCHS}  "
              f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"AUC={val_auc:.4f}  F1={val_f1:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_f1":  val_f1,
                "meta": meta,
            }, MODELS_DIR / "cnn_best.pt")
            print(f"   ✓ Best model saved (AUC={best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹ Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    checkpoint = torch.load(MODELS_DIR / "cnn_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    Xtest_t = torch.FloatTensor(X_test[:, None, :]).to(device)
    with torch.no_grad():
        test_logits = model(Xtest_t)
        test_probs  = torch.sigmoid(test_logits).cpu().numpy()
        test_preds  = (test_probs > 0.5).astype(int)

    metrics = _compute_metrics(y_test, test_preds, test_probs, "CNN+BiLSTM")
    _save_training_plots(history, "cnn")
    return metrics


# ─── Random Forest Training ────────────────────────────────────────────────────

def train_rf(features_df: pd.DataFrame, meta: dict) -> dict:
    print("\n🌲 Training Random Forest on HRV features")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = meta["feature_cols"]
    X = features_df[feature_cols].values.astype(np.float32)
    y = features_df["label"].values

    print(f"   Samples: {len(y)}  |  Features: {len(feature_cols)}")
    print(f"   Normal: {(y==0).sum()}  |  AFib: {(y==1).sum()}")

    # ── Patient-wise split ──
    train_idx, val_idx, test_idx, split_info = patient_wise_split(
        features_df, TEST_SIZE, VAL_SIZE
    )
    # Merge train+val for RF (RF doesn't need a separate val set — no early stopping)
    train_val_idx = np.concatenate([train_idx, val_idx])

    X_train = X[train_val_idx];  y_train = y[train_val_idx]
    X_test  = X[test_idx];       y_test  = y[test_idx]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "patient_split_rf.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"   Train+Val patients ({len(split_info['train_patients'])+len(split_info['val_patients'])}): "
          f"{sorted(split_info['train_patients'] + split_info['val_patients'])}")
    print(f"   Test patients ({len(split_info['test_patients'])}): "
          f"{sorted(split_info['test_patients'])}  ← HELD-OUT PATIENTS ONLY")

    # ── Clean features — replace inf/nan with column median ──
    # Some segments produce inf/nan HRV values (e.g. too few R-peaks,
    # division by zero in sd2, sample entropy edge cases)
    def clean_features(arr: np.ndarray, tag: str) -> np.ndarray:
        arr = arr.copy()
        n_inf = (~np.isfinite(arr)).sum()
        if n_inf > 0:
            print(f"   ⚠  {tag}: {n_inf} inf/nan values found — replacing with column median")
            for col in range(arr.shape[1]):
                mask = ~np.isfinite(arr[:, col])
                if mask.any():
                    median = np.nanmedian(arr[:, col][np.isfinite(arr[:, col])])
                    arr[mask, col] = median if np.isfinite(median) else 0.0
        return arr

    X_train = clean_features(X_train, "X_train")
    X_test  = clean_features(X_test,  "X_test")

    # Compute class weights
    class_counts = np.bincount(y_train)
    class_weight = {0: len(y_train) / (2 * class_counts[0]),
                    1: len(y_train) / (2 * class_counts[1])}
    print(f"   Class weights: {class_weight}")

    # SMOTE + RF Pipeline
    pipeline = ImbPipeline([
        ("smote",  SMOTE(sampling_strategy=SMOTE_RATIO,
                         random_state=RANDOM_STATE, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("   Fitting (SMOTE + RF)...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    metrics = _compute_metrics(y_test, test_preds, test_probs, "Random Forest")

    # Feature importance
    rf_model = pipeline.named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n   Top 5 features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"     {row['feature']:15s}  {row['importance']:.4f}")

    # Save pipeline
    joblib.dump(pipeline, MODELS_DIR / "rf_pipeline.pkl")
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    _save_feature_importance_plot(importance_df)

    print(f"\n✓ RF pipeline saved → {MODELS_DIR}/rf_pipeline.pkl")
    return metrics


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob, model_name: str) -> dict:
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ── Terminal results banner ──
    print(f"\n{'═'*55}")
    print(f"  ✅ {model_name} — BEST MODEL — Final Test Results")
    print(f"{'═'*55}")
    print(f"   AUC-ROC:     {auc:.4f}")
    print(f"   F1 Score:    {f1:.4f}")
    print(f"   Sensitivity: {rec:.4f}  ← AFib recall (critical!)")
    print(f"   Specificity: {spec:.4f}")
    print(f"   Precision:   {prec:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "AFib"],
                                 zero_division=0))

    # ── Confusion matrix printed to terminal ──
    print(f"   Confusion Matrix (on held-out test patients):")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │              Predicted               │")
    print(f"   │         Normal        AFib           │")
    print(f"   ├──────────┬────────────┬─────────────┤")
    print(f"   │ Normal   │  TN {tn:>6,}  │  FP {fp:>6,}   │")
    print(f"   │ AFib     │  FN {fn:>6,}  │  TP {tp:>6,}   │")
    print(f"   └──────────┴────────────┴─────────────┘")
    print(f"   Correctly identified AFib:  {tp:,} / {tp+fn:,} ({rec*100:.1f}%)")
    print(f"   Correctly cleared Normal:   {tn:,} / {tn+fp:,} ({spec*100:.1f}%)")
    print(f"   Missed AFib (false neg):    {fn:,}  ← patients told they're fine but aren't")
    print(f"   False alarms (false pos):   {fp:,}  ← normal patients flagged as AFib")

    # ── Save confusion matrix plot ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace("+", "_").replace(" ", "_")
    _save_confusion_matrix_plot(cm, model_name, safe_name)

    metrics = {
        "model": model_name, "auc": auc, "f1": f1,
        "sensitivity": rec, "specificity": spec,
        "precision": prec, "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "trained_at": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / f"metrics_{safe_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _save_confusion_matrix_plot(cm: np.ndarray, model_name: str, safe_name: str):
    """Save a clean confusion matrix plot to models/results/."""
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#050b12')
    ax.set_facecolor('#0c1620')

    # Draw cells manually for full control
    labels = [
        [f"TN\n{tn:,}\n(Correct Normal)", f"FP\n{fp:,}\n(False Alarm)"],
        [f"FN\n{fn:,}\n(Missed AFib!)",   f"TP\n{tp:,}\n(Caught AFib)"],
    ]
    colors = [
        ["#1a3a2a", "#3a1a1a"],
        ["#4a1a1a", "#1a3a4a"],
    ]
    text_colors = [
        ["#39ff6e", "#ff3b6b"],
        ["#ff3b6b", "#00e5ff"],
    ]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                         facecolor=colors[i][j], edgecolor='#1a2d3d', linewidth=2))
            ax.text(j + 0.5, 1 - i + 0.5, labels[i][j],
                    ha='center', va='center', fontsize=11,
                    color=text_colors[i][j], fontweight='bold',
                    linespacing=1.6)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAFib"],
                        color='#c8dde8', fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual\nAFib", "Actual\nNormal"],
                        color='#c8dde8', fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2d3d')

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ax.set_title(
        f"{model_name} — Confusion Matrix\n"
        f"Sensitivity {sens*100:.1f}%  |  Specificity {spec*100:.1f}%  |  "
        f"Test patients only",
        color='white', fontsize=10, pad=12
    )

    plt.tight_layout()
    out = RESULTS_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n   📊 Confusion matrix saved → {out}")


def _save_training_plots(history: dict, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#050b12')
    for ax in axes:
        ax.set_facecolor('#0c1620')
        ax.tick_params(colors='#c8dde8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2d3d')

    axes[0].plot(history["train_loss"], color='#00e5ff', label='Train')
    axes[0].plot(history["val_loss"],   color='#ff3b6b', label='Val')
    axes[0].set_title('Loss', color='white')
    axes[0].legend(facecolor='#0c1620', labelcolor='white')

    axes[1].plot(history["val_auc"], color='#39ff6e', label='AUC')
    axes[1].plot(history["val_f1"],  color='#ffb300', label='F1')
    axes[1].set_title('Validation Metrics', color='white')
    axes[1].set_ylim(0, 1)
    axes[1].legend(facecolor='#0c1620', labelcolor='white')

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / f"training_curves_{name}.png",
                dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📈 Training curves saved → {RESULTS_DIR}/training_curves_{name}.png")


def _save_feature_importance_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#050b12')
    ax.set_facecolor('#0c1620')
    colors = ['#00e5ff' if i < 5 else '#1a2d3d' for i in range(len(df))]
    ax.barh(df["feature"][::-1], df["importance"][::-1], color=colors[::-1])
    ax.set_title("HRV Feature Importance", color='white', fontsize=13)
    ax.tick_params(colors='#c8dde8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2d3d')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png",
                dpi=120, facecolor=fig.get_facecolor())
    plt.close()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "cnn", "both"],
                        default="both", help="Which model to train")
    args = parser.parse_args()

    # Load preprocessed data
    meta_path = PROCESSED_DIR / "meta.pkl"
    if not meta_path.exists():
        print("✗ No preprocessed data found. Run: python src/preprocess.py")
        sys.exit(1)

    meta = joblib.load(meta_path)
    print(f"✓ Loaded metadata: {meta['n_normal']} normal, {meta['n_afib']} AFib")

    results = {}

    if args.model in ("rf", "both"):
        features_path = PROCESSED_DIR / "hrv_features.csv"
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            results["rf"] = train_rf(features_df, meta)
        else:
            print("✗ hrv_features.csv not found — skip RF")

    if args.model in ("cnn", "both"):
        signals_path = PROCESSED_DIR / "signals.npy"
        labels_path  = PROCESSED_DIR / "labels.npy"
        if signals_path.exists() and labels_path.exists():
            signals = np.load(signals_path)
            labels  = np.load(labels_path)
            results["cnn"] = train_cnn(signals, labels, meta)
        else:
            print("✗ signals.npy not found — skip CNN")

    # Summary
    print("\n" + "═" * 50)
    print("  TRAINING COMPLETE — SUMMARY")
    print("═" * 50)
    for name, m in results.items():
        print(f"\n  {m.get('model', name).upper()}")
        print(f"    AUC:         {m.get('auc', 0):.4f}")
        print(f"    Sensitivity: {m.get('sensitivity', 0):.4f}  ← catch AFib")
        print(f"    Specificity: {m.get('specificity', 0):.4f}")
        print(f"    F1:          {m.get('f1', 0):.4f}")
    print("\n🎉 Ready for deployment: streamlit run app.py")


if __name__ == "__main__":
    main()