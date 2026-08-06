"""
Full metric table across all models: Accuracy / Precision / Recall (+ F1,
Balanced-Acc, ROC-AUC, PR-AUC). Each model is fit on TRAIN and scored on the
held-out VALIDATION set at the default 0.5 threshold (fair cross-model view;
TEST stays reserved). Positive class = DECLINED.
"""
from __future__ import annotations
import sys, time, warnings
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, average_precision_score)

import config as C
from data import make_splits
from pipeline import build_full_pipeline
from modeling import get_model_zoo, get_baselines

warnings.filterwarnings("ignore")
t0 = time.time()
s = make_splits()
Xtr, ytr = s["train"]; Xv, yv = s["val"]
yv = yv.values

rows = []
for name, est in {**get_baselines(), **get_model_zoo()}.items():
    try:
        pipe = build_full_pipeline(est).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xv)[:, 1]
        yp = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(yv, yp),
            "precision": precision_score(yv, yp, zero_division=0),
            "recall": recall_score(yv, yp, zero_division=0),
            "f1": f1_score(yv, yp, zero_division=0),
            "balanced_acc": balanced_accuracy_score(yv, yp),
            "roc_auc": roc_auc_score(yv, proba),
            "pr_auc": average_precision_score(yv, proba),
        })
        print(f"[{time.time()-t0:6.1f}s] {name:22s} acc={rows[-1]['accuracy']:.3f} "
              f"prec={rows[-1]['precision']:.3f} rec={rows[-1]['recall']:.3f}", flush=True)
    except Exception as e:
        print(f"[{time.time()-t0:6.1f}s] {name:22s} FAILED: {str(e)[:80]}", flush=True)

tbl = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
pd.set_option("display.width", 160)
print("\n=== ALL MODELS @ 0.5 threshold (val), ranked by PR-AUC ===")
print(tbl.round(4).to_string(index=False))
tbl.round(5).to_csv(C.REPORT_DIR / "model_metric_table.csv", index=False)
print(f"\n[written reports/model_metric_table.csv]  base DECLINED rate={yv.mean():.4f}")
