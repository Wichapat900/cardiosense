"""
evaluate.py — CardioSense Model Evaluation
===========================================
Generates full visual evaluation report matching the dashboard style.

Outputs:
  - models/results/report_rf.json
  - models/results/feature_importance.csv
  - models/results/evaluation_rf.png   ← full visual report

Usage:
  python src/evaluate.py
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import extract_features, FEATURE_NAMES

SAMPLE_RATE = 250
DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("models/results")
SAVED_DIR   = Path("models/saved")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#050b12"
PANEL    = "#080f18"
PANEL2   = "#0c1620"
BORDER   = "#1a2d3d"
TEXT     = "#c8dde8"
TEXT_MID = "#7a9bb8"
ACCENT   = "#2ab5b5"
SUCCESS  = "#1fcc7a"
DANGER   = "#f04060"
WARN     = "#f4a124"


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_MID,
        "axes.titlecolor":   TEXT,
        "xtick.color":       TEXT_MID,
        "ytick.color":       TEXT_MID,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "text.color":        TEXT,
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
    })


def load_test_data(bundle):
    signals     = np.load(DATA_DIR / "signals.npy")
    labels      = np.load(DATA_DIR / "labels.npy")
    patient_ids = np.load(DATA_DIR / "patient_ids.npy") \
                  if (DATA_DIR / "patient_ids.npy").exists() \
                  else np.arange(len(signals))

    gss  = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, temp_idx = next(gss.split(signals, labels, patient_ids))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    _, test_rel = next(gss2.split(
        signals[temp_idx], labels[temp_idx], patient_ids[temp_idx]))
    test_idx = temp_idx[test_rel]

    print(f"Evaluating on {len(test_idx)} test segments...")
    X_test, y_test = [], []
    for i, idx in enumerate(test_idx):
        if i % 1000 == 0:
            print(f"  {i}/{len(test_idx)}")
        feats = extract_features(signals[idx], SAMPLE_RATE)
        if feats is None:
            continue
        X_test.append([feats[f] for f in FEATURE_NAMES])
        y_test.append(int(labels[idx]))

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    imp    = SimpleImputer(strategy="median")
    X_test = imp.fit_transform(X_test).astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = bundle["scaler"].transform(X_test)
    return X_test, y_test


def plot_roc(ax, y_test, probs, auc):
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=BORDER, lw=1, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9,
              facecolor=PANEL2, edgecolor=BORDER, labelcolor=ACCENT)
    ax.grid(True, alpha=0.3)


def plot_pr(ax, y_test, probs, ap, baseline):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ax.plot(rec, prec, color=DANGER, lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color=BORDER, lw=1, linestyle="--",
               label=f"Baseline = {baseline:.3f}")
    ax.fill_between(rec, prec, alpha=0.08, color=DANGER)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=9,
              facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT_MID)
    ax.grid(True, alpha=0.3)


def plot_confusion(ax, cm, tn, fp, fn, tp):
    cmap = LinearSegmentedColormap.from_list("cs", [PANEL, ACCENT], N=256)
    ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "AFib"], color=TEXT)
    ax.set_yticklabels(["Normal", "AFib"], color=TEXT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    vals = [[tn, fp], [fn, tp]]
    clrs = [[SUCCESS, DANGER], [DANGER, SUCCESS]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{vals[i][j]:,}",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color=clrs[i][j])


def plot_threshold(ax, y_test, probs, best_thresh):
    thresholds = np.arange(0.10, 0.91, 0.01)
    sens, spec, f1s = [], [], []
    for t in thresholds:
        p  = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, p, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm.ravel()
        sens.append(tp_ / (tp_ + fn_ + 1e-8))
        spec.append(tn_ / (tn_ + fp_ + 1e-8))
        f1s.append(f1_score(y_test, p, zero_division=0))
    ax.plot(thresholds, sens, color=SUCCESS, lw=2, label="Sensitivity (Recall)")
    ax.plot(thresholds, spec, color=ACCENT,  lw=2, label="Specificity")
    ax.plot(thresholds, f1s,  color=WARN,    lw=2, label="F1 Score")
    ax.axvline(0.5,         color=TEXT_MID, lw=1, linestyle="--",
               label="Default threshold (0.5)", alpha=0.6)
    ax.axvline(best_thresh, color=WARN,     lw=1.5, linestyle="--",
               label=f"Best F1 threshold ({best_thresh:.2f})", alpha=0.9)
    ax.set_xlim([0.1, 0.9]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis — Tune sensitivity vs specificity for clinical use")
    ax.legend(fontsize=8, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT_MID, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_feature_importance(ax, rf, top_n=10):
    fi = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=True).tail(top_n)
    colors = [ACCENT if i >= top_n - 3 else TEXT_MID for i in range(len(fi))]
    bars = ax.barh(fi["feature"], fi["importance"],
                   color=colors, edgecolor=BORDER, height=0.6)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, fi["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=TEXT_MID)


def plot_metrics_box(ax, report):
    ax.axis("off")
    ax.set_facecolor(PANEL2)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    lines = [
        ("Random Forest",                                    TEXT,    11, "bold"),
        ("",                                                 TEXT_MID, 9, "normal"),
        (f"AUC-ROC:       {report['roc_auc']:.4f}",         ACCENT,  10, "normal"),
        (f"Avg Precision:  {report['avg_precision']:.4f}",  TEXT_MID, 9, "normal"),
        ("",                                                 TEXT_MID, 9, "normal"),
        (f"Sensitivity:   {report['sensitivity']:.4f}",     SUCCESS, 10, "normal"),
        (f"Specificity:   {report['specificity']:.4f}",     ACCENT,  10, "normal"),
        (f"Precision:     {report['precision']:.4f}",       TEXT_MID, 9, "normal"),
        (f"F1 Score:      {report['f1']:.4f}",              WARN,    10, "normal"),
        ("",                                                 TEXT_MID, 9, "normal"),
        (f"TP: {report['tp']}  FP: {report['fp']}",         SUCCESS,  9, "normal"),
        (f"FN: {report['fn']}  TN: {report['tn']}",         DANGER,   9, "normal"),
        ("",                                                 TEXT_MID, 9, "normal"),
        (f"Test set: {report['n_test']} samples",           TEXT_MID, 8, "normal"),
        (f"AFib prevalence: {report['afib_prev']:.1f}%",    TEXT_MID, 8, "normal"),
    ]
    y = 0.97
    for text, color, size, weight in lines:
        ax.text(0.05, y, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight=weight,
                verticalalignment="top", fontfamily="monospace")
        y -= 0.067


def evaluate():
    bundle_path = SAVED_DIR / "rf_pipeline.pkl"
    if not bundle_path.exists():
        print("No model found. Run: python src/train.py")
        return

    bundle = joblib.load(bundle_path)
    rf     = bundle["model"]
    thresh = bundle["threshold"]
    print(f"Loaded model — threshold={thresh:.2f}")

    set_dark_style()
    X_test, y_test = load_test_data(bundle)

    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= thresh).astype(int)
    cm    = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    auc      = float(roc_auc_score(y_test, probs))
    ap       = float(average_precision_score(y_test, probs))
    baseline = float(np.mean(y_test))
    prec_val = float(tp / (tp + fp + 1e-8))

    report = {
        "model":         "Random Forest (Explainable)",
        "n_features":    len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "threshold":     float(thresh),
        "n_test":        int(len(y_test)),
        "afib_prev":     float(np.mean(y_test) * 100),
        "roc_auc":       auc,
        "avg_precision": ap,
        "sensitivity":   float(tp / (tp + fn + 1e-8)),
        "specificity":   float(tn / (tn + fp + 1e-8)),
        "precision":     prec_val,
        "f1":            float(f1_score(y_test, preds, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    print(f"\n{'='*50}")
    print(f"  AUC-ROC:     {report['roc_auc']:.4f}")
    print(f"  Sensitivity: {report['sensitivity']:.4f}  <- AFib recall")
    print(f"  Specificity: {report['specificity']:.4f}")
    print(f"  Precision:   {report['precision']:.4f}")
    print(f"  F1:          {report['f1']:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*50}")

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle("CardioSense — Random Forest Evaluation Report",
                 fontsize=14, color=TEXT, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.42, wspace=0.35,
                           top=0.93, bottom=0.07,
                           left=0.06, right=0.97)

    plot_roc(fig.add_subplot(gs[0, 0]), y_test, probs, auc)
    plot_pr(fig.add_subplot(gs[0, 1]), y_test, probs, ap, baseline)
    plot_confusion(fig.add_subplot(gs[0, 2]), cm, tn, fp, fn, tp)
    plot_metrics_box(fig.add_subplot(gs[0, 3]), report)
    plot_threshold(fig.add_subplot(gs[1, 0:3]), y_test, probs, thresh)
    plot_feature_importance(fig.add_subplot(gs[1, 3]), rf, top_n=10)

    out_path = RESULTS_DIR / "evaluation_rf.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    with open(RESULTS_DIR / "report_rf.json", "w") as f:
        json.dump(report, f, indent=2)

    fi_df = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    print(f"\nSaved -> models/results/evaluation_rf.png")
    print(f"Saved -> models/results/report_rf.json")
    print(f"Saved -> models/results/feature_importance.csv")

    print(f"\nTop 5 features:")
    for _, row in fi_df.head(5).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")


if __name__ == "__main__":
    evaluate()