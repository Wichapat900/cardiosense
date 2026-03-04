"""
evaluate.py — CardioSense AFib Detection
=========================================
Generates full evaluation report:
  - ROC curve + AUC
  - Confusion matrix
  - Precision-Recall curve
  - Threshold sweep analysis
  - Per-record performance breakdown

Usage:
    python src/evaluate.py
    python src/evaluate.py --model rf
    python src/evaluate.py --model cnn
"""

import numpy as np
import pandas as pd
import joblib
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
RESULTS_DIR   = Path("models/results")

COLORS = {
    "bg":      "#050b12",
    "panel":   "#0c1620",
    "border":  "#1a2d3d",
    "text":    "#c8dde8",
    "accent":  "#00e5ff",
    "danger":  "#ff3b6b",
    "success": "#39ff6e",
    "warn":    "#ffb300",
}

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def styled_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(COLORS["bg"])
    axes_flat = [axes] if (nrows == 1 and ncols == 1) else np.array(axes).flatten()
    for ax in axes_flat:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
    return fig, axes


# ─── RF Evaluation ────────────────────────────────────────────────────────────

def evaluate_rf():
    pipeline_path = MODELS_DIR / "rf_pipeline.pkl"
    if not pipeline_path.exists():
        print("✗ RF model not found. Run: python src/train.py --model rf")
        return

    print("\n🌲 Evaluating Random Forest...")
    meta = joblib.load(PROCESSED_DIR / "meta.pkl")
    features_df = pd.read_csv(PROCESSED_DIR / "hrv_features.csv")

    X = features_df[meta["feature_cols"]].values.astype(np.float32)
    y = features_df["label"].values

    # Load the EXACT test patients train.py held out — no re-splitting
    split_path = RESULTS_DIR / "patient_split_rf.json"
    if not split_path.exists():
        print("✗ patient_split_rf.json not found.")
        print("  Re-run training first: python src/train.py --model rf")
        return

    with open(split_path) as f:
        split_info = json.load(f)

    test_patients = split_info["test_patients"]
    test_idx = features_df[features_df["record_id"].isin(test_patients)].index.to_numpy()
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Fix any NaN/Inf values from HRV feature computation
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   Test patients ({len(test_patients)}): {test_patients}")
    print(f"   Test segments: {len(y_test)}  ({int(y_test.sum())} AFib)")

    pipeline = joblib.load(pipeline_path)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    generate_report(y_test, y_pred, y_prob, "Random Forest", "rf")


# ─── CNN Evaluation ────────────────────────────────────────────────────────────

def evaluate_cnn():
    try:
        import torch
    except ImportError:
        print("✗ PyTorch not installed — skip CNN evaluation")
        return

    checkpoint_path = MODELS_DIR / "cnn_best.pt"
    if not checkpoint_path.exists():
        print("✗ CNN model not found. Run: python src/train.py --model cnn")
        return

    print("\n🧠 Evaluating CNN+BiLSTM...")
    from train import AFibCNNBiLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signals = np.load(PROCESSED_DIR / "signals.npy")
    labels  = np.load(PROCESSED_DIR / "labels.npy")

    # Load the EXACT test patients train.py held out — no re-splitting
    split_path = RESULTS_DIR / "patient_split_cnn.json"
    if not split_path.exists():
        print("✗ patient_split_cnn.json not found.")
        print("  Re-run training first: python src/train.py --model cnn")
        return

    with open(split_path) as f:
        split_info = json.load(f)

    # Reconstruct test indices from saved patient IDs
    features_df = pd.read_csv(PROCESSED_DIR / "hrv_features.csv")
    test_patients = split_info["test_patients"]
    test_idx = features_df[features_df["record_id"].isin(test_patients)].index.to_numpy()

    X_test = signals[test_idx]
    y_test = labels[test_idx]

    # Fix any NaN/Inf values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   Test patients ({len(test_patients)}): {test_patients}")
    print(f"   Test segments: {len(y_test)}  ({int(y_test.sum())} AFib)")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AFibCNNBiLSTM(input_len=signals.shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test[:, None, :])),
        batch_size=128, shuffle=False
    )

    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    generate_report(y_test, y_pred, y_prob, "CNN+BiLSTM", "cnn")


# ─── Report Generator ─────────────────────────────────────────────────────────

def generate_report(y_true, y_pred, y_prob, model_name: str, tag: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'═'*55}")
    print(classification_report(y_true, y_pred,
                                  target_names=["Normal", "AFib"],
                                  zero_division=0))

    # ── Plot 1: ROC + PR + Confusion Matrix + Threshold sweep ──
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=COLORS["accent"], lw=2,
             label=f"AUC = {roc_auc_val:.3f}")
    ax1.plot([0, 1], [0, 1], '--', color=COLORS["border"], lw=1)
    ax1.fill_between(fpr, tpr, alpha=0.1, color=COLORS["accent"])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.02])

    # Precision-Recall
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    ax2.plot(recall, precision, color=COLORS["danger"], lw=2,
             label=f"AP = {ap:.3f}")
    ax2.axhline(baseline, linestyle='--', color=COLORS["border"],
                label=f"Baseline = {baseline:.3f}")
    ax2.fill_between(recall, precision, alpha=0.1, color=COLORS["danger"])
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"])
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])

    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    _style_ax(ax3)
    cm = confusion_matrix(y_true, y_pred)
    im = ax3.imshow(cm, cmap='Blues', aspect='auto')
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Normal", "AFib"], color=COLORS["text"])
    ax3.set_yticklabels(["Normal", "AFib"], color=COLORS["text"])
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            # Diagonal = correct (white), off-diagonal = wrong (red)
            text_color = 'white' if i == j else COLORS["danger"]
            ax3.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                     color=text_color, fontsize=14, fontweight='bold')

    # Threshold sweep
    ax4 = fig.add_subplot(gs[1, 0:2])
    _style_ax(ax4)
    thresholds_sweep = np.linspace(0.1, 0.9, 80)
    sensitivities, specificities, f1s = [], [], []
    for t in thresholds_sweep:
        preds = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel() \
            if len(np.unique(preds)) > 1 else (0, 0, 0, 0)
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        f1s.append(f1_score(y_true, preds, zero_division=0))
    ax4.plot(thresholds_sweep, sensitivities, color=COLORS["success"],  lw=2, label="Sensitivity (Recall)")
    ax4.plot(thresholds_sweep, specificities, color=COLORS["accent"],   lw=2, label="Specificity")
    ax4.plot(thresholds_sweep, f1s,           color=COLORS["danger"],   lw=2, label="F1 Score")
    ax4.axvline(0.5, linestyle='--', color=COLORS["warn"], alpha=0.7, label="Default threshold (0.5)")
    best_f1_idx = np.argmax(f1s)
    ax4.axvline(thresholds_sweep[best_f1_idx], linestyle=':', color='white', alpha=0.5,
                label=f"Best F1 threshold ({thresholds_sweep[best_f1_idx]:.2f})")
    ax4.set_xlabel("Classification Threshold")
    ax4.set_ylabel("Score")
    ax4.set_title("Threshold Analysis — Tune sensitivity vs specificity for clinical use")
    ax4.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"], fontsize=8)
    ax4.set_xlim([0.1, 0.9]); ax4.set_ylim([0, 1.05])

    # Summary text
    ax5 = fig.add_subplot(gs[1, 2])
    _style_ax(ax5)
    ax5.axis('off')
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    summary = (
        f"  {model_name}\n\n"
        f"  AUC-ROC:      {roc_auc_val:.4f}\n"
        f"  Avg Precision:{ap:.4f}\n\n"
        f"  Sensitivity:  {sens:.4f}\n"
        f"  Specificity:  {spec:.4f}\n"
        f"  Precision:    {ppv:.4f}\n"
        f"  F1 Score:     {f1_score(y_true, y_pred, zero_division=0):.4f}\n\n"
        f"  TP: {tp}   FP: {fp}\n"
        f"  FN: {fn}   TN: {tn}\n\n"
        f"  Test set: {len(y_true)} samples\n"
        f"  AFib prevalence: {y_true.mean()*100:.1f}%"
    )
    ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             color=COLORS["text"],
             bbox=dict(facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                       boxstyle='round,pad=0.5'))

    fig.suptitle(f"CardioSense — {model_name} Evaluation Report",
                 color='white', fontsize=14, y=0.98)

    out_path = RESULTS_DIR / f"evaluation_{tag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊 Evaluation report saved → {out_path}")

    # Save JSON
    report_data = {
        "model": model_name,
        "roc_auc": float(roc_auc_val),
        "avg_precision": float(ap),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision_ppv": float(ppv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "best_f1_threshold": float(thresholds_sweep[best_f1_idx]),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "test_size": int(len(y_true)),
    }
    with open(RESULTS_DIR / f"report_{tag}.json", "w") as f:
        json.dump(report_data, f, indent=2)


def _style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "cnn", "both"], default="both")
    args = parser.parse_args()

    if args.model in ("rf", "both"):  evaluate_rf()
    if args.model in ("cnn", "both"): evaluate_cnn()
    print("\n✓ Evaluation complete. Check models/results/")


if __name__ == "__main__":
    main()