# EDA & Business Findings — ABC Credit Loan Approval

## Step 1 — Business framing

- Rows analysed (post-clean): **108,409**
- Target: **Decision** (APPROVED vs DECLINED). Positive class = **DECLINED** (the risky applicant we must catch).
- Problem type: **supervised binary classification** with probability output + business threshold.
- Base DECLINED rate: **5.26%**  → APPROVED:DECLINED ≈ **18.0:1** (severe imbalance).

**Cost asymmetry (why recall on DECLINED dominates):**
- *False approval* (we approve a truly risky applicant → predict APPROVED, actual DECLINED): direct credit loss (principal + collection cost); for an NBFC a single default can wipe the margin on ~15–20 good loans.
- *False rejection* (we decline a good applicant): opportunity cost = lost interest margin on one loan, plus mild CX damage.
- ⇒ **False approval ≫ false rejection in cost.** So we optimise **recall on the DECLINED class** (catch risky applicants) subject to keeping DECLINED precision high enough that we don't reject too much good business.
- **Accuracy is misleading**: predicting APPROVED for everyone scores ~94.7% accuracy while catching **0** risky applicants. We therefore rank models on **PR-AUC** and **ROC-AUC**, and tune the threshold on DECLINED recall/precision.

## Step 3 — Target distribution & imbalance

  -> figure: reports/figures/01_target_distribution.png
Interpretation: only **5.26%** of historical applications were declined. The model must not collapse to the majority class; class weighting / resampling and threshold tuning are mandatory, not optional.

## Numeric feature distributions (by outcome)

  -> figure: reports/figures/02_numeric_kde_by_target.png
  -> figure: reports/figures/03_violin_key_numerics.png
- **LTV**: median APPROVED=80.85 vs DECLINED=86.45  (Δ=+5.61).
- **Loan_Amount**: median APPROVED=89422.00 vs DECLINED=100808.00  (Δ=+11386.00).
- **Age**: median APPROVED=32.00 vs DECLINED=31.00  (Δ=-1.00).
- **Net_salary**: median APPROVED=30000.00 vs DECLINED=32000.00  (Δ=+2000.00).
- **income_to_loan**: median APPROVED=0.36 vs DECLINED=0.33  (Δ=-0.04).

Interpretation: **LTV** shows the clearest separation — declined applicants carry a **higher loan-to-value** (less down-payment / thinner collateral cushion). Loan_Amount is mildly higher for declines. Age and self-reported salary separate weakly, consistent with the univariate AUCs (LTV 0.74, Loan_Amount 0.61, Age 0.46, salary 0.51).

## Correlation among numeric features

  -> figure: reports/figures/04_correlation_heatmap.png
- Loan_Amount ↔ vehicle_value_est are highly correlated (value is derived from loan & LTV) — expected; trees are robust to it, and for linear models regularisation handles the redundancy.
- LTV ↔ Loan_Amount ≈ 0.45: bigger loans tend to have higher LTV. Otherwise features are fairly independent → each carries some non-redundant signal.

## Decline rate by segment (approval-rate analysis)

  -> figure: reports/figures/05_decline_rate_by_segment.png
Key segment effects (decline rate vs base 5.26%):
- **ltv_band**: 95+=19.4%, 85-95=10.4%, missing=6.2%, 75-85=4.5%, 60-75=1.4%, <=60=0.6%
- **PAST_LOANS_ACTIVE**: NO_PAST_LOANS=6.2%, PAST_LOANS_ACTIVE=4.2%
- **Product_Code**: MC=5.7%, SC=5.3%, MO=4.5%, EB=2.5%
- **Final_Tier**: 07 Rural=5.4%, 06 Semi-Urban=5.3%, 05 Urban=5.1%, 02 Megapolis (B)=4.8%, 04 Metro=4.1%, 03 Large Metro=3.4%, 01 Megapolis (A)=2.6%
- **Employment_Type**: NPP=6.9%, NREGI=5.6%, SAL=5.4%, SEP=5.2%, AGR=5.0%, STU=1.9%, NONEARNMEM=0.0%, PEN=0.0%

Interpretation:
- **ltv_band** is the strongest categorical lens: decline rate climbs steeply in the top LTV bucket — the single most actionable risk signal for the chatbot to collect.
- **PAST_LOANS_ACTIVE**: applicants with an active prior loan are declined *less* (existing repayment relationship / prior underwriting) — a useful, cheap-to-collect feature.
- **Product_Code**: electric (EB) applicants decline least; motorcycles (MC) most — product mix carries risk signal.
- **Final_Tier / geography**: rural & semi-urban decline slightly more than metro/megapolis.
- Demographic fields (Gender) are essentially flat vs base → weak predictor **and** a fairness liability; we keep it out of the headline drivers and monitor it for disparate impact.

## Missingness (post-clean, pre-impute)

- LTV: 0.060% missing
- Employment_Type: 0.023% missing
- Qualifications: 0.021% missing
- Net_salary: 0.021% missing
- Final_Tier: 0.020% missing
- Age: 0.016% missing
- Resident_Type: 0.016% missing
- Make_Code: 0.003% missing
- Model_Description: 0.003% missing
- Model_Variant: 0.003% missing
All missingness is <0.1% except engineered LTV-invalid→NaN and non-CC models (engine_cc). Handled by median/most-frequent imputation inside the pipeline (fit on train folds only).
