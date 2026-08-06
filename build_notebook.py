"""
Assemble ABC_Credit_Analysis.ipynb — the end-to-end analysis notebook
(cleaning -> EDA -> feature engineering -> feature selection -> baselines/model
zoo -> tuning -> evaluation -> final model).

Light steps run live; heavy results (13-model CV, tuning, final metrics, figures)
are loaded from the saved reports/ + artifacts/ so the notebook executes fast and
shows the true, authoritative numbers.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# --------------------------------------------------------------------------- #
md("""# ABC Credit — Loan Approval: End-to-End Analysis

Predict **APPROVED / DECLINED** for two-wheeler loan applications. This notebook
walks the full modelling pipeline — **data cleaning → EDA → feature engineering →
feature selection → baselines & model zoo → hyperparameter tuning → evaluation →
final calibrated model**.

**How to read this notebook**
- *Light* steps (cleaning, profiling, EDA, feature engineering) run **live** here.
- *Heavy* results (the 13-model 5-fold CV comparison, Optuna tuning, final test
  metrics, SHAP) are **loaded from the saved `reports/` + `artifacts/`** produced by
  the pipeline scripts (`02_model_selection.py`, `03_tune.py`, `04_final_eval.py`,
  `08_calibrate.py`) — so the numbers are the real ones without a 25-minute re-run.

**Headline:** severe class imbalance (5.26% DECLINED, ~18:1) + genuinely limited
signal (LTV univariate AUC ~0.74; all models plateau at ROC-AUC ~0.76–0.78). We
rank on **PR-AUC** and tune the **threshold for recall on the DECLINED class**.""")

code("""import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)

import config as C
from data import load_raw, get_clean_Xy, make_splits
from cleaning import clean_dataframe
from features import FeatureEngineer
from pipeline import build_preprocessor, build_full_pipeline

FIG = C.FIG_DIR
def show_fig(name):
    p = FIG / name
    display(Image(filename=str(p))) if p.exists() else print("missing:", p)
print("environment ready")""")

# ---- Step 1 -------------------------------------------------------------- #
md("""## 1 — Business framing

| | |
|---|---|
| Objective | approve good applicants fast, decline risky ones, minimise NBFC credit loss |
| Problem | supervised **binary classification** → probability + business threshold |
| Positive class | **DECLINED** (the risky applicant to catch) |
| Cost asymmetry | a **false approval** (fund a default) ≫ a **false rejection** (lose one good loan) |

**Accuracy is misleading** — "approve everyone" scores ~94.7% accuracy and catches
**zero** risky applicants. We rank models on **PR-AUC** and report **recall on
DECLINED** at a business threshold.""")

# ---- Step 2 -------------------------------------------------------------- #
md("## 2 — Load & inspect the raw data")
code("""raw = load_raw()
print("shape:", raw.shape)
raw.head()""")

code("""# target balance + missingness snapshot
tgt = raw["Decision"].value_counts(dropna=False)
print("Target distribution:\\n", tgt.to_string())
base = (raw["Decision"].str.upper() == "DECLINED").mean()
print(f"\\nDECLINED base rate: {base:.4%}  (~{(1-base)/base:.0f}:1 imbalance)")
print("\\nMissing % per column (top):")
print((raw.isna().mean()*100).round(3).sort_values(ascending=False).head(8).to_string())""")

# ---- Step 4 (quality) ---------------------------------------------------- #
md("""## 3 — Data-quality assessment (drives cleaning)

Issues surfaced by profiling — each maps to a cleaning rule.""")
code("""age = pd.to_numeric(raw["Age"], errors="coerce")
ltv = pd.to_numeric(raw["LTV"], errors="coerce")
sal = pd.to_numeric(raw["Net_salary"], errors="coerce")
mojibake = raw["Model_Description"].astype(str).str.contains(r"[^\\x00-\\x7F]", regex=True, na=False).sum()
print("Age: non-numeric tokens present ->", raw["Age"].notna().sum() - age.notna().sum(),
      "| ages > 60 (policy 18-60):", int((age > 60).sum()))
print("LTV == 0 (impossible):", int((ltv == 0).sum()), "| LTV > 100:", int((ltv > 100).sum()))
print("Net_salary == 0 (students/non-earners):", int((sal == 0).sum()))
print("Fully-duplicate rows:", int(raw.duplicated().sum()),
      "| dupes ignoring ID:", int(raw.duplicated(subset=[c for c in raw.columns if c!='ID']).sum()))
print("Mojibake rows in Model_Description (â€“ = mangled en-dash):", int(mojibake))""")

md("""**Cleaning rules (`src/cleaning.py`, stateless & leakage-free):** drop null-target
+ exact dupes; coerce `Age` (a stray `' '` token) → impute; `LTV<=0` → NaN → impute;
keep `Net_salary==0` as a valid non-earner signal; strip whitespace; **repair mojibake**
(latin-1→utf-8); keep ages 61–69 but flag them. Missingness is <0.1%/col → median /
most-frequent imputation **inside CV folds**.""")
code("""clean = clean_dataframe(raw, drop_null_target=True)
print("raw:", raw.shape, "-> clean:", clean.shape)
# proof: mojibake fixed, Age numeric, LTV zeros gone
print("mojibake rows after clean:",
      int(clean["Model_Description"].astype(str).str.contains(r"[^\\x00-\\x7F]", regex=True, na=False).sum()))
print("Age dtype:", clean["Age"].dtype, "| LTV==0 after clean:", int((clean["LTV"]==0).sum(skipna=True)))""")

# ---- Step 3 EDA ---------------------------------------------------------- #
md("""## 4 — Exploratory data analysis

Full figures live in `reports/figures/`; key numbers computed live below.""")
code("""X, y, _ = get_clean_Xy()
df = FeatureEngineer().fit_transform(X)
df["Decision"] = np.where(y == 1, "DECLINED", "APPROVED")

# LTV is the dominant driver — monotone decline gradient by band
band = (df.assign(_y=y.values).groupby("ltv_band")["_y"].agg(["mean","count"])
        .rename(columns={"mean":"decline_rate"}))
band["decline_rate"] = (band["decline_rate"]*100).round(1)
print("Decline rate by LTV band (%):"); print(band.to_string())

# univariate separation (AUC vs DECLINED)
from sklearn.metrics import roc_auc_score
print("\\nUnivariate AUC vs DECLINED:")
for c in ["LTV","Loan_Amount","Net_salary","Age"]:
    s = pd.to_numeric(df[c], errors="coerce").astype(float); m = s.notna()
    print(f"  {c:12s} {roc_auc_score(y[m], s[m]):.3f}")""")

md("Target imbalance, LTV separation, segment decline-rates, and the numeric correlation matrix:")
code("""show_fig("01_target_distribution.png")
show_fig("03_violin_key_numerics.png")
show_fig("05_decline_rate_by_segment.png")
show_fig("04_correlation_heatmap.png")""")

md("""**EDA takeaways:** LTV decline-rate is **monotone** (≤60 → 0.6% … 95+ → **19.4%**) —
the single strongest, most actionable risk signal. Self-reported salary is near-noise
(declines even have slightly higher median salary). `PAST_LOANS_ACTIVE` *lowers* decline;
electric vehicles decline least; rural > metro. `Gender` is flat → weak predictor **and**
a fairness liability (monitored, not a headline driver).""")

# ---- Step 6 FE ----------------------------------------------------------- #
md("""## 5 — Feature engineering (`src/features.py`)

Stateless, target-free transformer (safe inside CV & for single-row inference).""")
code("""fe_cols = ["income_to_loan","loan_to_income","vehicle_value_est","down_payment_est",
           "engine_cc","is_non_earner","high_ltv","past_loan_active","age_band","ltv_band",
           "salary_band","pincode_region"]
FeatureEngineer().fit_transform(X.head(5))[["LTV","Loan_Amount","Net_salary"]+fe_cols].head()""")
md("""`vehicle_value_est = Loan_Amount / (LTV/100)` reconstructs the asset price; `down_payment_est`
is skin-in-the-game; ratios capture repayment capacity/burden; flags mark non-earners, high-LTV,
policy-age; `pincode_region` = postal zone (unseen-safe, 9 levels vs 14k pincodes).""")

# ---- Step 7 feature selection ------------------------------------------- #
md("""## 6 — Feature selection (5 methods, `02_model_selection.py`)

Mutual information, ANOVA-F, χ², Random-Forest importance, and L1-logistic — averaged
into a cross-method rank. **Decision: keep all ~93 features** (tree models are robust;
pruning added fragility without lift). Ranking retained for interpretability.""")
code("""rank = pd.read_csv("reports/feature_ranking.csv", index_col=0)
cols = [c for c in ["mutual_info","anova_F","chi2","rf_importance","l1_abs_coef","mean_rank"] if c in rank.columns]
rank.sort_values("mean_rank", ascending=False)[cols].head(15).round(4)""")
code('show_fig("06_feature_importance.png")')

# ---- Step 8/9 pipeline + split ------------------------------------------ #
md("""## 7 — Preprocessing pipeline & split (`src/pipeline.py`, `src/data.py`)

`ColumnTransformer`: numeric (median impute + RobustScaler), binary passthrough,
ordinal tier, OHE (`handle_unknown='infrequent_if_exist'`), and TargetEncoder
(cross-fitted → leakage-safe) for high-cardinality `Pincode`/`Model_Description`.
Stratified 70/15/15 split; every transform is fit on **train folds only**.""")
code("""s = make_splits()
for name in ("train","val","test"):
    Xn, yn = s[name]; print(f"{name:5s} n={len(Xn):7d}  DECLINED={yn.mean():.4f}")
pipe = build_full_pipeline(__import__('sklearn.linear_model', fromlist=['LogisticRegression'])
                           .LogisticRegression(max_iter=1000, class_weight='balanced'))
pipe.fit(*s["train"])
print("\\nfeatures after preprocessing:", len(pipe.named_steps['pre'].get_feature_names_out()))
# unseen-category safety
Xbad = s["val"][0].head(3).copy()
for c,v in [("Make_Code","ZZ_NEW"),("Final_Tier","99 Moon Base"),("Pincode","999999")]:
    Xbad[c]=v
print("unseen categories predict without error:", pipe.predict_proba(Xbad)[:,1].round(3).tolist())""")

# ---- Steps 10-11 baselines + zoo ---------------------------------------- #
md("""## 8 — Baselines & model zoo (5-fold Stratified CV, ranked by PR-AUC)

13 models compared (`02_model_selection.py`). All configured for the ~18:1 imbalance
(`class_weight` / `scale_pos_weight`). Base rate = 0.0526.""")
code("""cv = pd.read_csv("reports/cv_results_table.csv")
show = [c for c in ["model","cv_pr_auc_mean","cv_pr_auc_std","cv_roc_auc_mean",
        "cv_recall_mean","cv_precision_mean","cv_f1_mean"] if c in cv.columns]
cv.sort_values("cv_pr_auc_mean", ascending=False)[show].round(4).reset_index(drop=True)""")
md("""**Every serious model plateaus at ROC-AUC ~0.76** — a real signal ceiling (LTV's
univariate AUC is 0.74). Gradient boosting leads on PR-AUC; a plain Logistic Regression
is strikingly close (and is retained as an interpretable challenger).""")

# ---- Step 12 tuning ------------------------------------------------------ #
md("""## 9 — Hyperparameter tuning (Optuna TPE, `03_tune.py`)

Top-2 families tuned to maximise 3-fold CV PR-AUC.""")
code("""tune = json.load(open("reports/tuning_results.json"))
for fam, d in tune.items():
    print(f"{fam}: best CV3 PR-AUC = {d['best_pr_auc_cv3']:.4f}")
    print("   params:", d["best_params"])""")

# ---- diagnostics / improvement ------------------------------------------ #
md("""## 10 — Did we push further? (diagnostics + squeeze + MLP)

`09_diagnostics.py`, `10–12`: learning curve, ablation, stacking, monotonic, and a
neural net — to test whether anything beats the tuned CatBoost.""")
code("""try:
    imp = json.load(open("reports/improvement_results.json"))
    print("Squeeze experiments (5-fold CV PR-AUC):")
    for r in imp["results"]:
        print(f"  {r['config']:18s} PR-AUC={r['pr_auc']:.4f}")
    print(f"  -> best={imp['best']['config']}  gain vs baseline={imp['gain_vs_baseline']:+.4f} "
          f"(CV noise ~{imp['cv_noise']:.4f}) -> adopt={imp['adopt']}")
except FileNotFoundError:
    print("improvement_results.json not present")
try:
    mlp = json.load(open("reports/mlp_results.json"))
    print("\\nNeural net (MLP):")
    for r in mlp: print(f"  {r['config']:16s} PR-AUC={r['pr_auc']:.4f} ROC-AUC={r['roc_auc']:.4f}")
except FileNotFoundError:
    pass""")
md("""**Verdict: keep the tuned CatBoost.** Learning curve is flat (data-saturated),
interactions/stacking/voting/MLP all land within CV noise. The ceiling is *information*
(features), not model — the real lever is new data (bureau score / verified income).""")
code('show_fig("14_learning_curve.png")')

# ---- Step 13 eval + threshold + calibration ----------------------------- #
md("""## 11 — Final model: evaluation, threshold & calibration

**Tuned CatBoost**, refit on train+val, evaluated once on the untouched **test set**.""")
code("""final = json.load(open("reports/final_metrics.json"))
meta = json.load(open("artifacts/model_metadata.json"))
m = final["test_chosen"]
rows = {k: m[k] for k in ["accuracy","precision_declined","recall_declined","f1_declined",
        "roc_auc","pr_auc","balanced_accuracy","specificity","mcc","log_loss"]}
print("Family:", final["family"], "| threshold:", final["threshold"])
display(pd.Series(rows).round(4).to_frame("test @ chosen threshold"))
cm = pd.DataFrame([[m["tn"],m["fp"]],[m["fn"],m["tp"]]],
                  index=["actual APPROVED","actual DECLINED"],
                  columns=["pred APPROVED","pred DECLINED"])
print("Confusion matrix:"); display(cm)""")

md("**Threshold trade-off (test) & calibration** — the decision cut-off is a business choice, not 0.5:")
code("""op = pd.read_csv("reports/test_operating_points.csv")
display(op.round(3))
cal = meta.get("calibration", {})
if cal:
    print(f"Calibration (isotonic): ECE {cal['ece_raw']:.3f} -> {cal['ece_calibrated']:.4f}, "
          f"Brier {cal['brier_raw']:.3f} -> {cal['brier_calibrated']:.3f}")
    print(f"Deployed rule: DECLINE if calibrated PD >= {cal['calibrated_pd_threshold']*100:.1f}%")""")
code("""show_fig("07_threshold_curves.png")
show_fig("13_calibration.png")
show_fig("09_shap_summary.png")""")

md("""## 12 — Final model summary

- **Model:** tuned, calibrated CatBoost. **Decision rule:** DECLINE if calibrated PD ≥ 6.9%.
- **Test:** ROC-AUC ≈ 0.78, PR-AUC ≈ 0.17 (3.3× the 0.053 base), recall on DECLINED ≈ 0.58–0.65.
- **Calibration:** ECE 0.35 → 0.003 — predicted PD ≈ observed bad rate.
- **Drivers (SHAP):** LTV ≫ Pincode > prior-loan ≈ Loan_Amount; self-reported income largely ignored.
- **Limits:** signal ceiling ~0.78 ROC on these 15 fields; label = historical decision (a proxy,
  not realised default). Next lever: bureau score + realised-default target + reject inference.

Full write-up: `reports/REPORT.md`. Chatbot/API: `reports/CHATBOT_PLAN.md`, `reports/PRODUCTION.md`.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "abc311", "display_name": "ABC (venv 3.11)"},
                  "language_info": {"name": "python"}}
with open("ABC_Credit_Analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"wrote ABC_Credit_Analysis.ipynb with {len(cells)} cells")
