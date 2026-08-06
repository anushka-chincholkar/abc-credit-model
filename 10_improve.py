"""
Squeeze experiments — can we beat the tuned-CatBoost baseline (PR-AUC ~0.159)?

Configs (all via 5-fold Stratified CV on TRAIN, identical folds):
  1. baseline           tuned CatBoost
  2. +interactions      InteractionEngineer (LTV², LTV×loan, ...)
  3. +monotonic         CatBoost with monotone constraints on LTV/down-payment/burden
  4. stacking           CatBoost + HistGB + LogReg  ->  LogReg meta-learner
  5. soft-voting        mean probability of the three

Adopt a config only if it beats baseline by more than the CV noise band.
"""
from __future__ import annotations
import sys, json, time, warnings
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config as C
from data import make_splits
from pipeline import build_full_pipeline, build_preprocessor
from features import FeatureEngineer, InteractionEngineer, INTERACTION_NUMERIC

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
def make_cat(**extra):
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed,
                              verbose=0, allow_writing_files=False, **cat_params, **extra)
def make_hgb():
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **hgb_params)
def make_lr():
    return LogisticRegression(max_iter=3000, class_weight="balanced")

def score(pipe, label):
    tm = time.time()
    pr = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    row = {"config": label, "pr_auc": float(pr.mean()), "pr_auc_std": float(pr.std()),
           "roc_auc": float(roc.mean()), "secs": round(time.time() - tm, 1)}
    print(f"[{time.time()-t0:6.1f}s] {label:16s} PR-AUC={row['pr_auc']:.4f}±{row['pr_auc_std']:.4f}  "
          f"ROC-AUC={row['roc_auc']:.4f}  ({row['secs']}s)", flush=True)
    return row

results = []

# 1. baseline
results.append(score(build_full_pipeline(make_cat()), "baseline_cat"))

# 2. interactions
results.append(score(
    build_full_pipeline(make_cat(), fe=InteractionEngineer(), extra_numeric=INTERACTION_NUMERIC),
    "cat+interactions"))

# 3. monotonic constraints -- build constraint vector from the (stable) feature order.
# Use HistGB's native monotonic_cst (clone-safe under sklearn CV, unlike CatBoost's
# list monotone_constraints). HistGB was tied with CatBoost, so it's a fair vehicle.
pre_probe = build_preprocessor()
pre_probe.fit(FeatureEngineer().fit_transform(X.head(3000)), y.head(3000))
names = list(pre_probe.get_feature_names_out())
MONO = {"num__LTV": 1, "bin__high_ltv": 1, "num__loan_to_income": 1,
        "num__down_payment_est": -1, "num__income_to_loan": -1}
mono_vec = np.array([MONO.get(n, 0) for n in names])
print(f"  monotone constraints set on: {[n for n in names if MONO.get(n,0)!=0]}", flush=True)
results.append(score(build_full_pipeline(make_hgb()), "histgb_baseline"))
hgb_mono = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed,
                                          monotonic_cst=mono_vec, **hgb_params)
results.append(score(build_full_pipeline(hgb_mono), "histgb+monotonic"))

# 4. stacking (shared preprocessing, stack raw estimators on the transformed matrix)
stack = Pipeline([
    ("fe", FeatureEngineer()),
    ("pre", build_preprocessor()),
    ("stack", StackingClassifier(
        estimators=[("cat", make_cat()), ("hgb", make_hgb()), ("lr", make_lr())],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        cv=3, n_jobs=1, passthrough=False)),
])
results.append(score(stack, "stacking"))

# 5. soft voting
vote = Pipeline([
    ("fe", FeatureEngineer()),
    ("pre", build_preprocessor()),
    ("vote", VotingClassifier(
        estimators=[("cat", make_cat()), ("hgb", make_hgb()), ("lr", make_lr())],
        voting="soft", n_jobs=1)),
])
results.append(score(vote, "soft_voting"))

# summary + adoption decision
tbl = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n=== RANKED ===", flush=True)
print(tbl.to_string(index=False), flush=True)
base = next(r for r in results if r["config"] == "baseline_cat")
best = tbl.iloc[0].to_dict()
gain = best["pr_auc"] - base["pr_auc"]
noise = base["pr_auc_std"]
adopt = best["config"] != "baseline_cat" and gain > noise
print(f"\nBest = {best['config']}  (Δ PR-AUC vs baseline = {gain:+.4f}; CV noise ≈ {noise:.4f})", flush=True)
print(f"Adoption decision: {'ADOPT ' + best['config'] if adopt else 'KEEP baseline (gain within noise -> ceiling confirmed)'}", flush=True)

out = {"results": results, "baseline_pr_auc": base["pr_auc"], "best": best,
       "gain_vs_baseline": gain, "cv_noise": noise, "adopt": bool(adopt)}
(C.REPORT_DIR / "improvement_results.json").write_text(json.dumps(out, indent=2))
print(f"\nDONE in {time.time()-t0:.0f}s -> reports/improvement_results.json", flush=True)
