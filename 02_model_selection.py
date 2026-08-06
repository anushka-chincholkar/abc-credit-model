"""
Step 7  : Feature-selection method comparison (MI / ANOVA / chi2 / tree / L1).
Step 10 : Baselines (Dummy, LogisticRegression).
Step 11 : Broad model zoo comparison via 5-fold Stratified CV.

Writes:
  reports/feature_ranking.csv
  reports/figures/06_feature_importance.png
  reports/cv_results.json          (incremental; safe to inspect mid-run)
Ranking metric = PR-AUC (average_precision): the honest headline for a 5% base
rate. ROC-AUC reported alongside.
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import config as C
from data import make_splits
from pipeline import build_full_pipeline, build_preprocessor
from features import FeatureEngineer
from modeling import get_model_zoo, get_baselines, CV_SCORING

warnings.filterwarnings("ignore")
t0 = time.time()

s = make_splits()
X_train, y_train = s["train"]
print(f"[{time.time()-t0:6.1f}s] train={len(X_train):,}  DECLINED={y_train.mean():.4f}", flush=True)

# =========================================================================== #
# STEP 7 : Feature-selection method comparison
# =========================================================================== #
print("\n=== STEP 7: feature-selection method comparison ===", flush=True)
# Fit preprocessing once to get a dense, named design matrix for the analysis.
fe = FeatureEngineer()
pre = build_preprocessor()
Xtr_fe = fe.fit_transform(X_train)
Xd = pre.fit_transform(Xtr_fe, y_train)
feat_names = list(pre.get_feature_names_out())
Xd = np.asarray(Xd, dtype=float)
print(f"design matrix: {Xd.shape}, {len(feat_names)} features", flush=True)

rank = pd.DataFrame(index=feat_names)
rank["mutual_info"] = mutual_info_classif(Xd, y_train, random_state=C.RANDOM_SEED)
F, _ = f_classif(Xd, y_train)
rank["anova_F"] = F
Xpos = MinMaxScaler().fit_transform(Xd)  # chi2 needs non-negative
chi, _ = chi2(Xpos, y_train)
rank["chi2"] = chi
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            min_samples_leaf=20, n_jobs=-1, random_state=C.RANDOM_SEED)
rf.fit(Xd, y_train)
rank["rf_importance"] = rf.feature_importances_
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                        class_weight="balanced", max_iter=2000)
l1.fit(Xd, y_train)
rank["l1_abs_coef"] = np.abs(l1.coef_.ravel())

# rank-average across methods (higher = more informative)
for col in rank.columns:
    rank[col + "_rk"] = rank[col].rank(ascending=True)
rank["mean_rank"] = rank[[c for c in rank.columns if c.endswith("_rk")]].mean(axis=1)
rank = rank.sort_values("mean_rank", ascending=False)
rank.to_csv(C.REPORT_DIR / "feature_ranking.csv")
print("Top 15 features by mean rank across 5 methods:", flush=True)
print(rank[["mutual_info", "anova_F", "chi2", "rf_importance", "l1_abs_coef", "mean_rank"]].head(15).round(4).to_string(), flush=True)

# figure: RF importance top-20
top = rank.sort_values("rf_importance", ascending=False).head(20)
plt.figure(figsize=(11, 9))
plt.barh(range(len(top))[::-1], top["rf_importance"].values, color="#2E86AB")
plt.yticks(range(len(top))[::-1], [t.split("__")[-1] for t in top.index], fontsize=10)
plt.xlabel("Random-Forest importance"); plt.title("Top-20 features (RF impurity importance)")
plt.tight_layout(); plt.savefig(C.FIG_DIR / "06_feature_importance.png", dpi=120); plt.close()
print("  -> reports/figures/06_feature_importance.png", flush=True)
print("Decision: keep all features. With only ~93 encoded columns and tree-based "
      "front-runners, pruning adds fragility without material lift; the ranking is "
      "reported for interpretability and to justify the top drivers.", flush=True)

# =========================================================================== #
# STEP 10 + 11 : baselines + zoo via 5-fold Stratified CV
# =========================================================================== #
print("\n=== STEP 10/11: baselines + model zoo (5-fold Stratified CV) ===", flush=True)
cv = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_SEED)
models = {**get_baselines(), **get_model_zoo()}
print("models:", list(models), flush=True)

results = []
out_path = C.REPORT_DIR / "cv_results.json"
for name, est in models.items():
    tm = time.time()
    try:
        pipe = build_full_pipeline(est)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING,
                                n_jobs=1, return_train_score=False, error_score="raise")
        row = {"model": name, "fit_seconds": round(time.time() - tm, 1)}
        for k in CV_SCORING:
            row[f"cv_{k}_mean"] = float(np.mean(scores["test_" + k]))
            row[f"cv_{k}_std"] = float(np.std(scores["test_" + k]))
        results.append(row)
        print(f"[{time.time()-t0:6.1f}s] {name:22s} "
              f"PR-AUC={row['cv_pr_auc_mean']:.4f}±{row['cv_pr_auc_std']:.4f}  "
              f"ROC-AUC={row['cv_roc_auc_mean']:.4f}  "
              f"F1={row['cv_f1_mean']:.4f}  ({row['fit_seconds']}s)", flush=True)
    except Exception as e:
        print(f"[{time.time()-t0:6.1f}s] {name:22s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
        results.append({"model": name, "error": str(e)[:300]})
    # incremental dump
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

# summary table
tbl = pd.DataFrame([r for r in results if "cv_pr_auc_mean" in r])
if len(tbl):
    tbl = tbl.sort_values("cv_pr_auc_mean", ascending=False)
    cols = ["model", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
            "cv_balanced_accuracy_mean", "cv_recall_mean", "cv_precision_mean",
            "cv_f1_mean", "fit_seconds"]
    print("\n=== RANKED BY CV PR-AUC ===", flush=True)
    print(tbl[cols].round(4).to_string(index=False), flush=True)
    tbl.to_csv(C.REPORT_DIR / "cv_results_table.csv", index=False)
print(f"\nDONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)
