import numpy as np
import json
from pathlib import Path

print("=" * 55)
print("CARDIOSENSE DATASET & MODEL STATS")
print("=" * 55)

sig_path = Path("data/processed/signals.npy")
lbl_path = Path("data/processed/labels.npy")
pid_path = Path("data/processed/patient_ids.npy")

if sig_path.exists():
    X = np.load(sig_path)
    y = np.load(lbl_path)
    g = np.load(pid_path) if pid_path.exists() else None
    print(f"\n--- PROCESSED DATA ---")
    print(f"Total segments:      {len(X):,}")
    print(f"AFib segments:       {np.sum(y==1):,}  ({np.mean(y==1)*100:.1f}%)")
    print(f"Normal segments:     {np.sum(y==0):,}  ({np.mean(y==0)*100:.1f}%)")
    if g is not None:
        print(f"Unique patients:     {len(np.unique(g))}")
else:
    print("data/processed/signals.npy not found")

rpt_path = Path("models/results/report_rf.json")
if rpt_path.exists():
    with open(rpt_path) as f:
        r = json.load(f)
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"AUC-ROC:        {r.get('roc_auc'):.4f}")
    print(f"Sensitivity:    {r.get('sensitivity'):.4f}")
    print(f"Specificity:    {r.get('specificity'):.4f}")
    print(f"F1 Score:       {r.get('f1'):.4f}")
    print(f"Threshold:      {r.get('threshold')}")
    print(f"TP={r.get('tp')}  FP={r.get('fp')}  FN={r.get('fn')}  TN={r.get('tn')}")
else:
    print("\nmodels/results/report_rf.json not found")

fi_path = Path("models/results/feature_importance.csv")
if fi_path.exists():
    import pandas as pd
    fi = pd.read_csv(fi_path)
    print(f"\n--- TOP 10 FEATURES ---")
    for _, row in fi.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

print("\n" + "=" * 55)