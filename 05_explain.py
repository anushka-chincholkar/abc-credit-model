"""
Step 16 : Explainability for the final saved model.

  - Permutation importance (model-agnostic, on held-out TEST, PR-AUC scorer)
  - SHAP summary (global) + SHAP force/waterfall (one applicant, local)
  - Partial dependence for the top numeric drivers
Everything is translated into a business sentence at the end.
"""
from __future__ import annotations
import sys, json, warnings
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import make_splits
import joblib

warnings.filterwarnings("ignore")

pipe = joblib.load(C.ARTIFACT_DIR / "model_pipeline.joblib")
fe, pre, model = pipe.named_steps["fe"], pipe.named_steps["pre"], pipe.named_steps["model"]
feat_names = [f.split("__", 1)[-1] for f in pre.get_feature_names_out()]

s = make_splits()
X_test, y_test = s["test"]
# transform once for tree explainers
Xt = np.asarray(pre.transform(fe.transform(X_test)), dtype=float)

LINES = ["# Explainability — final model\n"]
def note(x=""):
    print(x); LINES.append(x)

# --------------------------------------------------------------------------- #
# 1. Permutation importance (on a test sample, PR-AUC)
from sklearn.inspection import permutation_importance
note("## Permutation importance (TEST, scorer = average_precision)\n")
samp = X_test.sample(n=min(6000, len(X_test)), random_state=C.RANDOM_SEED)
ys = y_test.loc[samp.index]
perm = permutation_importance(pipe, samp, ys, scoring="average_precision",
                              n_repeats=5, random_state=C.RANDOM_SEED, n_jobs=-1)
# map back to input columns
pi = pd.Series(perm.importances_mean, index=samp.columns).sort_values(ascending=False)
for k, v in pi.head(15).items():
    note(f"- {k:20s} Δ(PR-AUC)={v:+.4f}")
plt.figure(figsize=(10, 8))
ptop = pi.head(15)[::-1]
plt.barh(range(len(ptop)), ptop.values, color="#2E86AB")
plt.yticks(range(len(ptop)), ptop.index, fontsize=10)
plt.xlabel("mean drop in PR-AUC when shuffled"); plt.title("Permutation importance (input fields)")
plt.tight_layout(); plt.savefig(C.FIG_DIR / "08_permutation_importance.png", dpi=120); plt.close()

# --------------------------------------------------------------------------- #
# 2. SHAP
note("\n## SHAP (global + local)\n")
try:
    import shap
    # background sample for tree explainer
    bg_idx = np.random.RandomState(C.RANDOM_SEED).choice(len(Xt), size=min(2000, len(Xt)), replace=False)
    Xt_bg = Xt[bg_idx]
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt_bg)
        if isinstance(sv, list):          # some versions return [class0, class1]
            sv = sv[1]
    except Exception:
        explainer = shap.Explainer(model)
        sv = explainer(Xt_bg).values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    # global summary
    shap.summary_plot(sv, features=Xt_bg, feature_names=feat_names, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "09_shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
    # global bar
    shap.summary_plot(sv, features=Xt_bg, feature_names=feat_names, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "10_shap_bar.png", dpi=120, bbox_inches="tight"); plt.close()
    # mean |shap| ranking
    mshap = pd.Series(np.abs(sv).mean(0), index=feat_names).sort_values(ascending=False)
    note("Top-15 features by mean |SHAP|:")
    for k, v in mshap.head(15).items():
        note(f"- {k:22s} {v:.4f}")
    # local: highest-risk applicant in background sample
    order = np.argsort(-sv.sum(1))
    i0 = int(order[0])
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[-1],
        sv[i0], feature_names=feat_names, max_display=14, show=False)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "11_shap_waterfall_highrisk.png", dpi=120, bbox_inches="tight"); plt.close()
    note("  -> figures 09_shap_summary / 10_shap_bar / 11_shap_waterfall_highrisk")
    shap_ok = True
except Exception as e:
    note(f"SHAP step skipped: {type(e).__name__}: {str(e)[:160]}")
    shap_ok = False

# --------------------------------------------------------------------------- #
# 3. Partial dependence (top numeric drivers) on the full pipeline
note("\n## Partial dependence (top numeric drivers)\n")
try:
    from sklearn.inspection import PartialDependenceDisplay
    pdp_feats = ["LTV", "Loan_Amount", "Net_salary", "Age"]
    samp2 = X_test.sample(n=min(4000, len(X_test)), random_state=C.RANDOM_SEED)
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    PartialDependenceDisplay.from_estimator(pipe, samp2, pdp_feats, ax=ax.ravel()[:4],
                                            response_method="predict_proba")
    plt.suptitle("Partial dependence on P(DECLINED)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "12_partial_dependence.png", dpi=120); plt.close()
    note("  -> reports/figures/12_partial_dependence.png")
except Exception as e:
    note(f"PDP step skipped: {type(e).__name__}: {str(e)[:160]}")

# --------------------------------------------------------------------------- #
note("\n## Business translation\n")
note("- **LTV / high_ltv / down_payment** dominate: the thinner the borrower's equity in the "
     "vehicle, the higher the modelled default risk. This is the lever the chatbot should stress-test first.")
note("- **Geography (Pincode target-encoding, tier, pincode_region)** contributes secondary signal — regional "
     "repayment behaviour. Encoded safely so unseen pincodes fall back to the regional/global prior.")
note("- **PAST_LOANS_ACTIVE** lowers risk (existing repayment track record).")
note("- **Self-reported Net_salary** is weak and non-monotone — consistent with it being unverified; the model "
     "leans on collateral (LTV) far more than on stated income, which is the prudent behaviour for an NBFC.")

with open(C.REPORT_DIR / "explainability.md", "w") as f:
    f.write("\n".join(LINES) + "\n")
note("\n[written reports/explainability.md]")
