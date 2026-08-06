"""
Model diagnostics — is the ceiling data (features) or model?

  A. Learning curve: CV PR-AUC vs training size. Flat tail => data-saturated,
     i.e. more rows won't help; the limit is the features.
  B. Feature ablation: drop each business concept, measure CV PR-AUC loss.
  C. Bayes-error estimate: kNN label disagreement in the preprocessed feature
     space -> a rough floor on irreducible error and an implied max achievable AUC.
"""
from __future__ import annotations
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import config as C
from data import make_splits
from pipeline import build_full_pipeline, build_preprocessor
from features import FeatureEngineer

t0 = time.time()
seed = C.RANDOM_SEED
s = make_splits()
X_train, y_train = s["train"]

with open(C.REPORT_DIR / "tuning_results.json") as f:
    cat_params = json.load(f)["CatBoost"]["best_params"]
from catboost import CatBoostClassifier
def make_cat():
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed,
                              verbose=0, allow_writing_files=False, **cat_params)

cv3 = StratifiedKFold(3, shuffle=True, random_state=seed)
def cvprauc(pipe, X, y):
    return cross_val_score(pipe, X, y, cv=cv3, scoring="average_precision", n_jobs=1)

OUT = {}

# ---------------------------------------------------------------- A. curve
print("=== A. LEARNING CURVE ===", flush=True)
sizes = [5000, 15000, 30000, 50000, X_train.shape[0]]
curve = []
for n in sizes:
    if n < len(X_train):
        Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=n, stratify=y_train, random_state=seed)
    else:
        Xs, ys = X_train, y_train
    sc = cvprauc(build_full_pipeline(make_cat()), Xs.reset_index(drop=True), ys.reset_index(drop=True))
    curve.append({"n": int(n), "pr_auc_mean": float(sc.mean()), "pr_auc_std": float(sc.std())})
    print(f"  n={n:6d}  PR-AUC={sc.mean():.4f} ± {sc.std():.4f}   ({time.time()-t0:.0f}s)", flush=True)
OUT["learning_curve"] = curve
plt.figure(figsize=(9, 6))
xs = [c["n"] for c in curve]; ys_ = [c["pr_auc_mean"] for c in curve]; es = [c["pr_auc_std"] for c in curve]
plt.errorbar(xs, ys_, yerr=es, marker="o", color="#2E86AB", capsize=4)
plt.xlabel("training rows"); plt.ylabel("CV PR-AUC"); plt.title("Learning curve (tuned CatBoost)")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(C.FIG_DIR / "14_learning_curve.png", dpi=120); plt.close()
delta = curve[-1]["pr_auc_mean"] - curve[-2]["pr_auc_mean"]
print(f"  last doubling Δ PR-AUC = {delta:+.4f}  -> {'PLATEAU (data-saturated)' if abs(delta)<0.005 else 'still climbing'}", flush=True)

# ---------------------------------------------------------------- B. ablation
print("\n=== B. FEATURE ABLATION (drop one concept group) ===", flush=True)
GROUPS = {
    "LTV/collateral": ["LTV"],
    "geography": ["Pincode", "Final_Tier"],
    "prior_loan": ["PAST_LOANS_ACTIVE"],
    "loan_amount": ["Loan_Amount"],
    "income": ["Net_salary"],
    "vehicle": ["Product_Code", "Make_Code", "Model_Description", "Model_Variant"],
    "demographics": ["Gender", "Age", "Qualifications", "Resident_Type"],
}
base_sc = cvprauc(build_full_pipeline(make_cat()), X_train, y_train)
base_mean = float(base_sc.mean())
print(f"  baseline (all features)      PR-AUC={base_mean:.4f}", flush=True)
abl = {"baseline": base_mean}
for name, cols in GROUPS.items():
    # neutralise (set NaN -> imputed to a constant) rather than drop, so the
    # ColumnTransformer schema stays intact but the group carries no signal.
    Xd = X_train.copy()
    for c in cols:
        if c in Xd.columns:
            Xd[c] = np.nan
    sc = cvprauc(build_full_pipeline(make_cat()), Xd, y_train)
    abl[name] = float(sc.mean())
    print(f"  drop {name:16s}  PR-AUC={sc.mean():.4f}  Δ={sc.mean()-base_mean:+.4f}", flush=True)
OUT["ablation"] = abl

# ---------------------------------------------------------------- C. Bayes error
print("\n=== C. BAYES-ERROR / MAX-AUC ESTIMATE (kNN label disagreement) ===", flush=True)
# preprocess a sample, look at each point's k neighbours' labels
from sklearn.neighbors import NearestNeighbors
samp_idx = np.random.RandomState(seed).choice(len(X_train), size=min(15000, len(X_train)), replace=False)
Xs = X_train.iloc[samp_idx]; ys = y_train.iloc[samp_idx].values
pre = build_preprocessor()
Xd = np.asarray(pre.fit_transform(FeatureEngineer().fit_transform(Xs), ys), dtype=float)
# standardise for distance
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9)
k = 50
nn = NearestNeighbors(n_neighbors=k + 1).fit(Xd)
_, idx = nn.kneighbors(Xd)
neigh = ys[idx[:, 1:]]                      # drop self
local_bad_rate = neigh.mean(1)
# Bayes error ~ average of min(p, 1-p) over local neighbourhoods
bayes_err = float(np.mean(np.minimum(local_bad_rate, 1 - local_bad_rate)))
# Implied max AUC via neighbourhood-probability ranking (optimistic proxy)
from sklearn.metrics import roc_auc_score
try:
    implied_auc = float(roc_auc_score(ys, local_bad_rate))
except Exception:
    implied_auc = float("nan")
OUT["bayes"] = {"k": k, "bayes_error_estimate": bayes_err,
                "implied_max_auc_local": implied_auc, "base_rate": float(ys.mean())}
print(f"  base rate={ys.mean():.4f}  est. Bayes error≈{bayes_err:.4f}  "
      f"implied local max AUC≈{implied_auc:.4f}", flush=True)
print("  (interpretation: if implied max AUC is close to our 0.78, the fields are "
      "near-exhausted; higher would mean a better model could still extract more.)", flush=True)

with open(C.REPORT_DIR / "diagnostics.json", "w") as f:
    json.dump(OUT, f, indent=2)
print(f"\nDONE diagnostics in {time.time()-t0:.0f}s -> reports/diagnostics.json", flush=True)
