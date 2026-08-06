"""
Neural net (MLP) evaluation — does deep learning beat GBTs on this tabular task?

Configs (5-fold Stratified CV on TRAIN, PR-AUC / ROC-AUC):
  * mlp_plain        : fe -> pre -> StandardScaler -> MLP  (MLP has no class_weight;
                       ranking metrics are threshold-free so this is a fair rank test)
  * mlp_oversampled  : same, with RandomOverSampler inside the CV fold (handles the
                       5% imbalance during training)
  * stack_with_mlp   : CatBoost + HistGB + LogReg + MLP -> LogReg meta (does an NN,
                       which errs differently from trees, help the ensemble?)
MLPs need every feature on a comparable scale, so we StandardScale the full
transformed matrix (OHE/ordinal/target-encoded/robust-scaled numerics alike).
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import config as C
from data import make_splits
from pipeline import build_preprocessor
from features import FeatureEngineer

warnings.filterwarnings("ignore")
t0 = time.time()
seed = C.RANDOM_SEED
s = make_splits()
X, y = s["train"]
cv = StratifiedKFold(5, shuffle=True, random_state=seed)

tuning = json.loads((C.REPORT_DIR / "tuning_results.json").read_text())
cat_params = tuning["CatBoost"]["best_params"]
hgb_params = tuning["HistGradientBoosting"]["best_params"]


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3,
                         batch_size=256, learning_rate_init=1e-3, max_iter=120,
                         early_stopping=True, n_iter_no_change=10, random_state=seed)


def score(pipe, label):
    tm = time.time()
    pr = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    row = {"config": label, "pr_auc": float(pr.mean()), "pr_auc_std": float(pr.std()),
           "roc_auc": float(roc.mean()), "secs": round(time.time() - tm, 1)}
    print(f"[{time.time()-t0:6.1f}s] {label:18s} PR-AUC={row['pr_auc']:.4f}±{row['pr_auc_std']:.4f}  "
          f"ROC-AUC={row['roc_auc']:.4f}  ({row['secs']}s)", flush=True)
    return row


results = []

# 1. plain MLP
mlp_plain = Pipeline([("fe", FeatureEngineer()), ("pre", build_preprocessor()),
                      ("sc", StandardScaler()), ("mlp", make_mlp())])
results.append(score(mlp_plain, "mlp_plain"))

# 2. oversampled MLP (imblearn pipeline keeps resampling inside the fold)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
mlp_os = ImbPipeline([("fe", FeatureEngineer()), ("pre", build_preprocessor()),
                      ("sc", StandardScaler()),
                      ("ros", RandomOverSampler(random_state=seed)), ("mlp", make_mlp())])
results.append(score(mlp_os, "mlp_oversampled"))

# 3. stack with MLP as a diverse base learner
from catboost import CatBoostClassifier
def make_cat():
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed,
                              verbose=0, allow_writing_files=False, **cat_params)
def make_hgb():
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **hgb_params)

stack_mlp = Pipeline([
    ("fe", FeatureEngineer()), ("pre", build_preprocessor()), ("sc", StandardScaler()),
    ("stack", StackingClassifier(
        estimators=[("cat", make_cat()), ("hgb", make_hgb()),
                    ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                    ("mlp", make_mlp())],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3, n_jobs=1))])
results.append(score(stack_mlp, "stack_with_mlp"))

tbl = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n=== RANKED (baseline CatBoost = 0.1619) ===", flush=True)
print(tbl.to_string(index=False), flush=True)
(C.REPORT_DIR / "mlp_results.json").write_text(json.dumps(results, indent=2))
print(f"\nDONE in {time.time()-t0:.0f}s -> reports/mlp_results.json", flush=True)
