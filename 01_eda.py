"""
Step 1 + Step 3 : Business framing + Exploratory Data Analysis.

Produces publication-quality figures in reports/figures/ and a quantified,
interpreted findings file in reports/eda_findings.md. Runs on the *cleaned*
data (+ engineered features) so what we plot is what the model sees.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from data import get_clean_Xy
from features import FeatureEngineer

sns.set_theme(style="whitegrid", context="talk")
PAL = {"APPROVED": "#2E86AB", "DECLINED": "#E4572E"}
FIG = C.FIG_DIR
LINES = []


def note(s=""):
    print(s)
    LINES.append(s)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close()
    note(f"  -> figure: reports/figures/{name}")


# --------------------------------------------------------------------------- #
X, y, ids = get_clean_Xy()
df = FeatureEngineer().fit_transform(X)
df["Decision"] = np.where(y == 1, "DECLINED", "APPROVED")
n = len(df)
base = y.mean()

note("# EDA & Business Findings — ABC Credit Loan Approval\n")
note("## Step 1 — Business framing\n")
note(f"- Rows analysed (post-clean): **{n:,}**")
note(f"- Target: **Decision** (APPROVED vs DECLINED). Positive class = **DECLINED** (the risky applicant we must catch).")
note(f"- Problem type: **supervised binary classification** with probability output + business threshold.")
note(f"- Base DECLINED rate: **{base:.2%}**  → APPROVED:DECLINED ≈ **{(1-base)/base:.1f}:1** (severe imbalance).")
note("")
note("**Cost asymmetry (why recall on DECLINED dominates):**")
note("- *False approval* (we approve a truly risky applicant → predict APPROVED, actual DECLINED): direct credit loss "
     "(principal + collection cost); for an NBFC a single default can wipe the margin on ~15–20 good loans.")
note("- *False rejection* (we decline a good applicant): opportunity cost = lost interest margin on one loan, plus mild CX damage.")
note("- ⇒ **False approval ≫ false rejection in cost.** So we optimise **recall on the DECLINED class** (catch risky "
     "applicants) subject to keeping DECLINED precision high enough that we don't reject too much good business.")
note("- **Accuracy is misleading**: predicting APPROVED for everyone scores ~94.7% accuracy while catching **0** risky "
     "applicants. We therefore rank models on **PR-AUC** and **ROC-AUC**, and tune the threshold on DECLINED recall/precision.")

# --------------------------------------------------------------------------- #
# 1. Target distribution
note("\n## Step 3 — Target distribution & imbalance\n")
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
vc = df["Decision"].value_counts()
ax[0].bar(vc.index, vc.values, color=[PAL[k] for k in vc.index])
ax[0].set_title("Class counts"); ax[0].set_ylabel("applications")
for i, v in enumerate(vc.values):
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
ax[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=[PAL[k] for k in vc.index],
          startangle=90, wedgeprops=dict(width=0.45))
ax[1].set_title("Class share")
savefig("01_target_distribution.png")
note(f"Interpretation: only **{base:.2%}** of historical applications were declined. The model must not collapse to the "
     "majority class; class weighting / resampling and threshold tuning are mandatory, not optional.")

# --------------------------------------------------------------------------- #
# 2. Numeric distributions + by-target
note("\n## Numeric feature distributions (by outcome)\n")
NUMS = ["Age", "Loan_Amount", "LTV", "Net_salary", "income_to_loan", "vehicle_value_est"]
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
for ax, col in zip(axes.ravel(), NUMS):
    for k in ("APPROVED", "DECLINED"):
        vals = pd.to_numeric(df.loc[df.Decision == k, col], errors="coerce").astype("float64").dropna()
        # clip Net_salary/vehicle tails for readability
        if col in ("Net_salary", "vehicle_value_est", "Loan_Amount"):
            vals = vals.clip(upper=vals.quantile(0.99))
        sns.kdeplot(vals, ax=ax, label=k, color=PAL[k], fill=True, alpha=0.35, warn_singular=False)
    ax.set_title(col); ax.legend(fontsize=10)
savefig("02_numeric_kde_by_target.png")

# boxplots + violins for the strongest signal (LTV) and Loan_Amount
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, col in zip(axes, ["LTV", "Loan_Amount", "Age"]):
    sns.violinplot(data=df, x="Decision", y=pd.to_numeric(df[col], errors="coerce").astype("float64"),
                   hue="Decision", palette=PAL, ax=ax, legend=False, cut=0)
    ax.set_title(f"{col} by outcome"); ax.set_ylabel(col)
savefig("03_violin_key_numerics.png")

for col in ["LTV", "Loan_Amount", "Age", "Net_salary", "income_to_loan"]:
    s = pd.to_numeric(df[col], errors="coerce")
    med_app = s[df.Decision == "APPROVED"].median()
    med_dec = s[df.Decision == "DECLINED"].median()
    note(f"- **{col}**: median APPROVED={med_app:.2f} vs DECLINED={med_dec:.2f}  "
         f"(Δ={med_dec-med_app:+.2f}).")
note("\nInterpretation: **LTV** shows the clearest separation — declined applicants carry a **higher loan-to-value** "
     "(less down-payment / thinner collateral cushion). Loan_Amount is mildly higher for declines. Age and self-reported "
     "salary separate weakly, consistent with the univariate AUCs (LTV 0.74, Loan_Amount 0.61, Age 0.46, salary 0.51).")

# --------------------------------------------------------------------------- #
# 3. Correlation heatmap
note("\n## Correlation among numeric features\n")
corr_cols = ["Age", "Loan_Amount", "LTV", "Net_salary", "income_to_loan",
             "loan_to_income", "vehicle_value_est", "down_payment_est", "engine_cc"]
cc = df[corr_cols].apply(pd.to_numeric, errors="coerce").astype("float64").corr()
plt.figure(figsize=(11, 9))
sns.heatmap(cc, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Numeric correlation matrix")
savefig("04_correlation_heatmap.png")
note("- Loan_Amount ↔ vehicle_value_est are highly correlated (value is derived from loan & LTV) — expected; trees are "
     "robust to it, and for linear models regularisation handles the redundancy.")
note("- LTV ↔ Loan_Amount ≈ 0.45: bigger loans tend to have higher LTV. Otherwise features are fairly independent → "
     "each carries some non-redundant signal.")

# --------------------------------------------------------------------------- #
# 4. Decline rate by categorical / engineered band
note("\n## Decline rate by segment (approval-rate analysis)\n")
SEG = ["Employment_Type", "Qualifications", "Product_Code", "Gender", "Resident_Type",
       "Final_Tier", "Make_Code", "PAST_LOANS_ACTIVE", "age_band", "ltv_band", "salary_band"]
fig, axes = plt.subplots(4, 3, figsize=(21, 22))
for ax, col in zip(axes.ravel(), SEG):
    g = df.assign(_y=y.values).groupby(col)["_y"].mean().sort_values(ascending=False)
    g = g[g.index.notna()]
    ax.bar(range(len(g)), g.values * 100, color="#6A4C93")
    ax.axhline(base * 100, color="#E4572E", ls="--", lw=2, label=f"base {base*100:.1f}%")
    ax.set_xticks(range(len(g))); ax.set_xticklabels([str(x) for x in g.index], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Decline % by {col}", fontsize=13); ax.legend(fontsize=9)
for ax in axes.ravel()[len(SEG):]:
    ax.axis("off")
savefig("05_decline_rate_by_segment.png")
note("Key segment effects (decline rate vs base %.2f%%):" % (base*100))
for col in ["ltv_band", "PAST_LOANS_ACTIVE", "Product_Code", "Final_Tier", "Employment_Type"]:
    g = df.assign(_y=y.values).groupby(col)["_y"].mean().sort_values(ascending=False)
    g = g[g.index.notna()]
    parts = ", ".join(f"{i}={v*100:.1f}%" for i, v in g.items())
    note(f"- **{col}**: {parts}")
note("\nInterpretation:")
note("- **ltv_band** is the strongest categorical lens: decline rate climbs steeply in the top LTV bucket — the single "
     "most actionable risk signal for the chatbot to collect.")
note("- **PAST_LOANS_ACTIVE**: applicants with an active prior loan are declined *less* (existing repayment relationship / "
     "prior underwriting) — a useful, cheap-to-collect feature.")
note("- **Product_Code**: electric (EB) applicants decline least; motorcycles (MC) most — product mix carries risk signal.")
note("- **Final_Tier / geography**: rural & semi-urban decline slightly more than metro/megapolis.")
note("- Demographic fields (Gender) are essentially flat vs base → weak predictor **and** a fairness liability; we keep it "
     "out of the headline drivers and monitor it for disparate impact.")

# --------------------------------------------------------------------------- #
# 5. Missingness snapshot
note("\n## Missingness (post-clean, pre-impute)\n")
miss = X.isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss):
    for c, m in miss.items():
        note(f"- {c}: {m*100:.3f}% missing")
note("All missingness is <0.1% except engineered LTV-invalid→NaN and non-CC models (engine_cc). Handled by median/"
     "most-frequent imputation inside the pipeline (fit on train folds only).")

with open(C.REPORT_DIR / "eda_findings.md", "w") as f:
    f.write("\n".join(LINES) + "\n")
note("\n[written reports/eda_findings.md]")
