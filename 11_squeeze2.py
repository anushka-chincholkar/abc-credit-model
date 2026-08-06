"""
Squeeze experiments (robust re-run) + Bayes-error ceiling.

Handles the two infrastructure issues found:
  * CatBoost list monotone_constraints can't be sklearn-cloned  -> use HistGB.
  * OHE(min_frequency) yields a *variable* column count across folds -> a fixed
    monotonic_cst mismatches. Fix: a wrapper that builds the constraint vector at
    fit() time from X.shape[1], constraining only the FIXED-position leading
    numeric/binary features (LTV, down-payment, burden, high_ltv); OHE/target-enc
    columns (which come last and vary) get 0.
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import config as C
from data import make_splits
from pipeline import build_full_pipeline, build_preprocessor
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

from catboost import CatBoostClassifier
def make_cat():
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed,
                              verbose=0, allow_writing_files=False, **cat_params)
def make_hgb():
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **hgb_params)
def make_lr():
    return LogisticRegression(max_iter=3000, class_weight="balanced")


# Fixed leading-position constraints (ColumnTransformer order: numeric[0..8],
# binary[9..12], ...). LTV=2(+), income_to_loan=4(-), loan_to_income=5(+),
# down_payment_est=7(-), high_ltv=11(+).
LEADING_CST = {2: 1, 4: -1, 5: 1, 7: -1, 11: 1}


class LeadingMonotonicHGB(BaseEstimator, ClassifierMixin):
    """HistGB whose monotonic_cst is built at fit-time to match X width."""
    def __init__(self, cst_by_index=None, **hgb_kwargs):
        self.cst_by_index = cst_by_index or {}
        self.hgb_kwargs = hgb_kwargs

    def fit(self, X, y):
        n = X.shape[1]
        cst = np.zeros(n, dtype=int)
        for i, v in self.cst_by_index.items():
            if i < n:
                cst[i] = v
        self.model_ = HistGradientBoostingClassifier(monotonic_cst=cst, **self.hgb_kwargs)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


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
results.append(score(build_full_pipeline(make_cat()), "baseline_cat"))

# monotonic (robust wrapper)
hgb_mono = LeadingMonotonicHGB(cst_by_index=LEADING_CST, class_weight="balanced",
                               random_state=seed, **hgb_params)
results.append(score(build_full_pipeline(hgb_mono), "histgb+monotonic"))

# stacking (shared preprocessing, meta LR)
stack = Pipeline([("fe", FeatureEngineer()), ("pre", build_preprocessor()),
                  ("stack", StackingClassifier(
                      estimators=[("cat", make_cat()), ("hgb", make_hgb()), ("lr", make_lr())],
                      final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                      cv=3, n_jobs=1))])
results.append(score(stack, "stacking"))

# soft voting
vote = Pipeline([("fe", FeatureEngineer()), ("pre", build_preprocessor()),
                 ("vote", VotingClassifier(
                     estimators=[("cat", make_cat()), ("hgb", make_hgb()), ("lr", make_lr())],
                     voting="soft", n_jobs=1))])
results.append(score(vote, "soft_voting"))

tbl = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n=== RANKED ===", flush=True)
print(tbl.to_string(index=False), flush=True)
base = next(r for r in results if r["config"] == "baseline_cat")
best = tbl.iloc[0].to_dict()
gain = best["pr_auc"] - base["pr_auc"]
adopt = best["config"] != "baseline_cat" and gain > base["pr_auc_std"]
print(f"\nBest={best['config']}  Δvs baseline={gain:+.4f}  CV-noise≈{base['pr_auc_std']:.4f}  "
      f"-> {'ADOPT' if adopt else 'KEEP baseline (within noise)'}", flush=True)

# ---------------- Bayes-error / max-AUC ceiling ---------------- #
print("\n=== BAYES-ERROR / MAX-AUC ESTIMATE (kNN label disagreement) ===", flush=True)
from sklearn.neighbors import NearestNeighbors
idx = np.random.RandomState(seed).choice(len(X), size=min(15000, len(X)), replace=False)
Xs, ys = X.iloc[idx], y.iloc[idx].values
pre = build_preprocessor()
Xd = np.asarray(pre.fit_transform(FeatureEngineer().fit_transform(Xs), ys), dtype=float)
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9)
k = 50
_, nidx = NearestNeighbors(n_neighbors=k + 1).fit(Xd).kneighbors(Xd)
local = ys[nidx[:, 1:]].mean(1)
bayes_err = float(np.mean(np.minimum(local, 1 - local)))
implied_auc = float(roc_auc_score(ys, local))
print(f"  base rate={ys.mean():.4f}  Bayes error≈{bayes_err:.4f}  implied local max AUC≈{implied_auc:.4f}", flush=True)

out = {"results": results, "best": best, "gain_vs_baseline": gain,
       "cv_noise": base["pr_auc_std"], "adopt": bool(adopt),
       "bayes": {"k": k, "bayes_error": bayes_err, "implied_max_auc": implied_auc,
                 "base_rate": float(ys.mean())}}
(C.REPORT_DIR / "improvement_results.json").write_text(json.dumps(out, indent=2))
print(f"\nDONE in {time.time()-t0:.0f}s -> reports/improvement_results.json", flush=True)
