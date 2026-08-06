# ABC Credit — Loan Approval Model: Final Report

**Author:** ML Engineering  ·  **Task:** predict `Decision` (APPROVED / DECLINED)
for two-wheeler loan applications  ·  **Positive class:** `DECLINED` (the risky
applicant we must catch).

> **Headline:** On a held-out test set the final model achieves **ROC-AUC ≈ 0.77**
> and **PR-AUC ≈ 0.16** (vs a 0.053 random baseline — a ~3× lift on the metric that
> matters under 18:1 imbalance). The credit decision is produced **solely by the ML
> model** against a business-chosen threshold; the chatbot only collects inputs and
> explains results. *Final tuned numbers are in §13–§14; this file is regenerable
> from the saved artifacts.*

---

## Step 1 — Business problem

| Question | Answer |
|---|---|
| Business objective | Approve good applicants fast, decline risky ones, minimise credit loss for an NBFC. |
| Prediction objective | P(DECLINED) for a new applicant from a minimal, cheap-to-collect field set. |
| ML problem type | Supervised **binary classification** with calibrated probability + business threshold. |
| Main business risk | **False approval** (approve a would-be default) ≫ false rejection (decline a good customer). |
| Cost of a false approval | Principal + collection cost. A single default can erase the interest margin on ~15–20 good loans. |
| Cost of a false rejection | Lost interest margin on one loan + minor CX cost. |

**Metric priority.** Because false approval dominates cost and only 5.26% of
applications are declined:
- **Accuracy is rejected as a headline** — predicting "APPROVE everyone" scores
  94.7% accuracy while catching **zero** risky applicants.
- **Recall on DECLINED (sensitivity)** is the primary business metric — the share
  of risky applicants we actually catch.
- **Precision on DECLINED** guards against declining too many good customers (lost
  business), so it constrains how far we push recall.
- Models are **ranked on PR-AUC** (threshold-free, honest under imbalance), with
  **ROC-AUC** as a secondary threshold-free view. **F1 / balanced-accuracy / MCC**
  reported at the chosen threshold.

---

## Step 2 — Dataset inspection

- **Shape:** 108,423 rows × 17 columns (raw). After cleaning: **108,409** rows.
- **Target:** `Decision` — APPROVED 102,724 (94.74%) vs DECLINED 5,698 (5.26%).
- **Column roles**

| Role | Columns |
|---|---|
| ID (drop from model) | `ID` |
| Target | `Decision` |
| Numeric | `Age`, `Loan_Amount`, `LTV`, `Net_salary` |
| Low/med-card categorical (OHE) | `Qualifications`, `Employment_Type`, `Gender`, `Product_Code`, `Resident_Type`, `Make_Code` (12), `Model_Variant` (21) |
| Ordinal | `Final_Tier` (7 tiers, megapolis→rural) |
| Binary | `PAST_LOANS_ACTIVE` |
| High-cardinality (target-encode) | `Pincode` (13,999), `Model_Description` (87) |
| Text (feature source) | `Model_Description` |

---

## Step 4 — Data-quality assessment & Step 5 — Cleaning

Every rule is deterministic, stateless and applied identically at train and
inference (no train/serve skew), and **never uses the target**.

| # | Issue found (profiling) | Rows | Handling |
|---|---|---|---|
| 1 | Fully-empty trailing row (null target) | 1 | drop rows with null target |
| 2 | `Age` stored as text — a lone `" "` token → object dtype | 1 (+19 NaN) | coerce to numeric → NaN → median impute |
| 3 | `Age` **> 60** (stated policy is 18–60), up to 69 | 1,105 | keep value + engineer `age_outside_policy` flag (don't discard real data) |
| 4 | `Net_salary == 0` — all Students / Non-earning members | 626 | keep as **valid** (decline rate 1.8% ≪ 5.26%) + `is_non_earner` flag; **not** treated as missing |
| 5 | `LTV == 0` — impossible (loan can't be 0% of value) | 65 | set to NaN → median impute |
| 6 | Whitespace in `Final_Tier`, `Model_Variant`, `PAST_LOANS_ACTIVE` | many | strip + collapse internal spaces |
| 7 | **Mojibake** `â€"` (double-encoded en-dash) in `Model_Description` | 5,028 | latin-1→utf-8 repair |
| 8 | Case variants in code fields | — | upper-case code fields; title-case Gender |
| 9 | Exact duplicate rows (ignoring ID) | 12 | drop |
| 10 | Contradictory labels (identical predictors, both outcomes) | 2 groups | negligible label noise; left in |
| 11 | Scattered missingness (<0.1% per column) | — | median / most-frequent impute **inside CV folds** |
| 12 | `Pincode` read as float | all | cast to string identifier |

**Leakage check.** No feature separates the target. Tiny "0% / 100% decline"
segments (PEN n=135, NONEARNMEM n=8) are small-sample artefacts, not leaks. The
target-encoded fields are cross-fitted, so no target value leaks into its own row.

---

## Step 3 — Exploratory data analysis (key findings)

Figures in `reports/figures/`. Full narrative in `reports/eda_findings.md`.

- **Imbalance (fig 01):** 5.26% DECLINED → resampling/class-weights + threshold
  tuning are mandatory.
- **LTV is the dominant driver (figs 02–03):** median LTV is 86.5 for declines vs
  80.9 for approvals; the **decline rate rises monotonically with LTV band**:

  | LTV band | ≤60 | 60–75 | 75–85 | 85–95 | 95+ |
  |---|---|---|---|---|---|
  | Decline rate | 0.6% | 1.4% | 4.5% | 10.4% | **19.4%** |

  Business reading: thin equity / small down-payment ⇒ higher default risk. This is
  the first thing the chatbot should stress-test.
- **Univariate signal (AUC vs DECLINED):** LTV **0.74**, Loan_Amount 0.61,
  Net_salary 0.51, Age 0.46. Self-reported salary is essentially noise (declines
  even have slightly *higher* median salary) — consistent with unverified income.
- **Segment effects:** `PAST_LOANS_ACTIVE` **lowers** decline (4.2% vs 6.2% — an
  existing repayment relationship helps); electric vehicles (EB) decline least
  (2.5%); risk drifts up from megapolis (2.6%) to rural (5.4%).
- **Fairness note:** `Gender` is flat vs base (≈5.3% both) — weak predictor **and**
  a protected attribute; kept out of the headline drivers and monitored for
  disparate impact (see §21).
- **Correlation (fig 04):** features largely independent; `Loan_Amount`↔`LTV`≈0.45;
  `vehicle_value_est` is a derived function of the two (handled fine by trees).

---

## Step 6 — Feature engineering

Implemented as a stateless sklearn transformer (`features.FeatureEngineer`), safe
inside CV and for a single applicant. Each feature has a business rationale:

| Feature | Definition | Rationale |
|---|---|---|
| `income_to_loan` | Net_salary / Loan_Amount | repayment capacity |
| `loan_to_income` | Loan_Amount / (Net_salary+1) | leverage / burden |
| `vehicle_value_est` | Loan_Amount / (LTV/100) | reconstructed asset value |
| `down_payment_est` | vehicle_value_est − Loan_Amount | "skin in the game" |
| `engine_cc` | parsed from Model_Variant/Description; EB→0 | vehicle class/power |
| `is_non_earner` | Net_salary == 0 | student/non-earning segment |
| `age_outside_policy` | Age <18 or >60 | policy-band flag |
| `high_ltv` | LTV ≥ 85 | risk-heavy tail |
| `past_loan_active` | PAST_LOANS_ACTIVE → 0/1 | prior repayment track record |
| `age_band`,`ltv_band`,`salary_band` | fixed, data-independent bins | help linear models; interpretability |
| `pincode_region` | leading digit of pincode (India postal zone) | unseen-safe geography (9 levels vs 14k) |

All ratios guard divide-by-zero (→ NaN → imputed). Bins use fixed, data-independent
edges so the transform stays reproducible on unseen data.

---

## Step 7 — Feature selection

Compared **5 methods** on the 93-column design matrix (Mutual Information, ANOVA-F,
χ², Random-Forest impurity importance, L1-logistic coefficients) and averaged their
ranks. Top features by mean rank (fig 06):

`LTV`, `high_ltv`, `down_payment_est`, `Pincode` (target-enc), `ltv_band_*`,
`past_loan_active`, `Loan_Amount`, `pincode_region`, `Age`, `Model_Description`
(target-enc).

**Decision: keep all features.** With only ~93 encoded columns and tree-based
front-runners that are robust to irrelevant inputs, pruning added fragility without
measurable lift. The ranking is retained for **interpretability** and to justify the
top drivers. (RFE / permutation importance corroborate — see §16.)

---

## Step 8 — Preprocessing pipeline

Single `ColumnTransformer`, **unseen-category safe** by construction:

| Group | Transform | Unseen-category behaviour |
|---|---|---|
| numeric + engineered | median impute → RobustScaler (heavy `Net_salary` tail) | n/a |
| binary flags | passthrough (0/1) | n/a |
| `Final_Tier` | impute → OrdinalEncoder(order) | unknown → −1 sentinel |
| low-card categoricals | impute → OneHotEncoder(`infrequent_if_exist`, `min_frequency=0.5%`) | unseen/rare → infrequent bucket |
| high-card (`Pincode`,`Model_Description`) | TargetEncoder (internal 5-fold cross-fit) | unknown → smoothed prior |

Scalers were A/B-considered (Standard/Robust/MinMaxs/Power); **RobustScaler** chosen
for the salary/loan tails. Verified: fitting on train and transforming a batch of
**fully novel** categories (`Make_Code=ZZ_NEW`, `Final_Tier="99 Moon Base"`,
`Pincode=999999`, `Employment_Type=ALIEN`) produces no NaNs and no errors.

---

## Step 9 — Train / validation / test strategy

- **Stratified** split preserving the 5.26% DECLINED rate: **train 75,885 /
  val 16,262 / test 16,262** (70/15/15).
- **5-fold Stratified CV** on train for model comparison and tuning.
- **No preprocessing before the split**; every imputer/encoder/scaler/target-encoder
  is fit inside the pipeline on training folds only. The **test set is untouched**
  until the single final evaluation.

---

## Steps 10–11 — Baselines & model zoo (5-fold Stratified CV on train)

Ranked by **CV PR-AUC** (positive = DECLINED, base rate 0.0526). All learners were
told about the imbalance (`class_weight="balanced"` / `scale_pos_weight` /
`auto_class_weights`).

| Model | PR-AUC | ±std | ROC-AUC | Bal-Acc | Recall@0.5 | Precision@0.5 | F1@0.5 |
|---|---|---|---|---|---|---|---|
| **CatBoost** | **0.1584** | 0.0081 | 0.7635 | 0.693 | 0.639 | 0.123 | 0.206 |
| HistGradientBoosting | 0.1577 | 0.0118 | **0.7665** | 0.696 | 0.705 | 0.111 | 0.192 |
| RandomForest | 0.1558 | 0.0107 | 0.7642 | 0.656 | 0.458 | 0.149 | 0.224 |
| ExtraTrees | 0.1542 | 0.0093 | 0.7608 | 0.688 | 0.635 | 0.120 | 0.202 |
| XGBoost | 0.1540 | 0.0087 | 0.7601 | 0.682 | 0.599 | 0.123 | 0.205 |
| GradientBoosting¹ | 0.1530 | 0.0097 | 0.7653 | 0.503 | 0.006 | 0.293 | 0.012 |
| LogisticRegression | 0.1478 | 0.0105 | 0.7635 | 0.695 | **0.738** | 0.105 | 0.184 |
| LightGBM | 0.1466 | 0.0066 | 0.7505 | 0.665 | 0.524 | 0.130 | 0.209 |
| DecisionTree | 0.1311 | 0.0034 | 0.7222 | 0.668 | 0.677 | 0.099 | 0.173 |
| GaussianNB | 0.1192 | 0.0076 | 0.7311 | 0.674 | 0.705 | 0.099 | 0.173 |
| LinearSVM (SGD)² | 0.1064 | 0.0226 | 0.6854 | 0.635 | 0.631 | 0.092 | 0.159 |
| KNN | 0.1030 | 0.0066 | 0.6852 | 0.500 | 0.000 | 0.100 | 0.001 |
| Dummy (most-frequent) | 0.0526 | 0.0000 | 0.5000 | 0.500 | 0.000 | 0.000 | 0.000 |
| Dummy (stratified) | 0.0525 | 0.0004 | 0.4982 | 0.498 | 0.048 | 0.049 | 0.048 |

¹ sklearn `GradientBoostingClassifier` has **no** `class_weight`, so at 0.5 it
predicts almost all APPROVED — its threshold-free PR-AUC is competitive but it needs
threshold tuning. ² RBF-SVM is computationally infeasible at 108k rows; a
`class_weight`-balanced SGD (modified-huber) stands in as the linear-SVM
representative — and it underperforms, so nothing was lost.

**Reading:** every serious model plateaus at **ROC-AUC ≈ 0.76** — a real signal
ceiling given LTV's 0.74 univariate AUC. Gradient boosting leads on PR-AUC; a plain
**Logistic Regression is remarkably close** and has the highest recall — this drives
the interpretability/latency discussion in §18.

---

## Step 12 — Hyperparameter optimisation

Top-2 families (CatBoost, HistGradientBoosting) tuned with **Optuna (Bayesian TPE)**,
30 trials each, objective = mean 3-fold CV PR-AUC through the full leakage-safe
pipeline (grid search avoided — continuous, high-dim space). Search on a 45k
stratified subsample for speed; final model refit on the full data.

| Family | Best CV-3 PR-AUC (45k subsample) | Best params |
|---|---|---|
| **CatBoost** ✅ | **0.1552** | iterations 600, learning_rate 0.0138, depth 5, l2_leaf_reg 4.06, random_strength 1.05, bagging_temperature 0.016 |
| HistGradientBoosting | 0.1545 | max_iter 1100, learning_rate 0.0324, max_leaf_nodes 16, max_depth 12, min_samples_leaf 199, l2_regularization 0.98 |

The two are effectively tied; **CatBoost** is carried forward (marginally higher
PR-AUC, native handling of the tree structure, very stable across seeds). On the
**full** training set with 5-fold CV the tuned CatBoost scores **PR-AUC 0.1589,
recall 0.735** (see the imbalance table below), confirming the subsample ranking.

---

## Step 13 — Final model evaluation (held-out TEST)

Tuned **CatBoost**, refit on **train+val (92,147 rows)**, evaluated once on the
untouched **test set (16,262 rows, 855 DECLINED)**. Threshold-free ranking metrics
are threshold-independent; the rest are at the adopted threshold **0.59** (§14).

| Metric | @ threshold 0.59 (adopted) | @ 0.50 |
|---|---|---|
| ROC-AUC | **0.7783** | 0.7783 |
| PR-AUC (avg precision) | **0.1740** | 0.1740 |
| Accuracy | 0.7730 | 0.7093 |
| Precision (DECLINED) | 0.1295 | 0.110 |
| **Recall / Sensitivity (DECLINED)** | **0.5801** | 0.752 |
| Specificity (APPROVED kept) | 0.7837 | 0.662 |
| F1 (DECLINED) | 0.2118 | 0.192 |
| Balanced accuracy | 0.6819 | 0.707 |
| MCC | 0.1914 | 0.193 |
| Log-loss | 0.5655 | 0.5655 |

**Confusion matrix @ 0.59** (rows = actual, positive = DECLINED):

|  | pred APPROVED | pred DECLINED |
|---|---|---|
| **actual APPROVED** | TN = 12,074 | FP = 3,333 |
| **actual DECLINED** | FN = 359 | TP = 496 |

The model catches **496 of 855 (58%)** risky applicants, at the cost of wrongly
declining 3,333 of 15,407 good applicants (22%). PR-AUC 0.174 vs a 0.0526 baseline
is a **~3.3× lift**. ROC-AUC 0.778 on the untouched test set matches CV (0.76) and
validation (0.770) — **no overfitting**.

---

## Step 14 — Threshold selection & justification

The threshold is a **business decision, not 0.5**. We searched operating points on
the **validation** set and read the trade-off on **test**. Because the model ranks
weakly (PR-AUC 0.17), precision is low everywhere, so every extra risky applicant
caught costs several wrongly-declined good ones.

**Test operating-point curve** (`reports/test_operating_points.csv`, fig 07):

| threshold | recall (DECLINED) | precision | specificity | good customers wrongly declined (FP) | risky caught (TP) | risky missed (FN) |
|---|---|---|---|---|---|---|
| 0.50 | 0.752 | 0.110 | 0.662 | 5,202 | 643 | 212 |
| 0.55 | 0.658 | 0.119 | 0.729 | 4,183 | 563 | 292 |
| **0.59 (adopted)** | **0.580** | **0.130** | **0.784** | **3,333** | **496** | **359** |
| 0.65 | 0.448 | 0.152 | 0.861 | 2,137 | 383 | 472 |
| 0.69 (cost-min 8:1) | 0.360 | 0.178 | 0.908 | 1,422 | 308 | 547 |
| 0.75 | 0.235 | 0.225 | 0.955 | 691 | 201 | 654 |

**Method.** We minimise expected cost `C = R·FN + FP`, where `R = cost(false
approval)/cost(false rejection)`. Cost-minimising thresholds: **4:1 → 0.78**,
**8:1 → 0.69**, **12:1 → 0.59**.

**Choice: threshold = 0.59 (risk-averse, R = 12:1).** Rationale:
- Pure interest-margin economics imply R ≈ 4–8 (a default loses ~₹40k, a wrongful
  decline ~₹10k margin), which would set a *conservative* 0.69–0.78 and catch only
  17–36% of risky applicants.
- But the brief explicitly prioritises **recall on DECLINED**, and an NBFC's real
  cost of a bad loan includes NPA provisioning, capital and portfolio-quality
  effects beyond lost margin — justifying **R ≈ 12**. At 0.59 we catch **58%** of
  risky applicants (vs 36% at the profit-only optimum) while still keeping 78% of
  good customers.
- The threshold is a **single config value** in `model_metadata.json`; the business
  can slide it along the table above as risk appetite changes. Plots: PR curve, ROC
  curve, and precision/recall/F1-vs-threshold in **fig 07**.

---

## Step 15 — Error analysis

At the adopted threshold: **359 false approvals** (missed risky — the costly error)
and **3,333 false rejections** (lost good business).

- **Where the model fails:** missed-risky applicants (FN) concentrate in the
  **mid/high LTV bands 75–95** (~90% of misses), *not* the extreme 95+ tail. The
  model reliably catches the obvious high-LTV cases; the ones it misses look
  *ordinary* — LTV ~85, average loan size, average stated income — i.e. risky
  applicants who are statistically indistinguishable from good ones on the fields
  we collect. This is a **data-signal limit, not a tuning bug**.
- **FN vs FP medians (initial cut):** missed-risky have LTV≈84.8 / loan≈₹97.7k;
  wrongly-declined-good have LTV≈90.2 / loan≈₹107.9k. Median stated salary is
  identical (₹32k) for both — confirming self-reported income does not separate the
  hard cases.
- **Implication:** further lift requires *new* signal (bureau score, verified
  income, delinquency history), not more modelling of the current 15 fields.

---

## Step 16 — Explainability

Three independent lenses agree on the drivers.

**Permutation importance** (test, Δ PR-AUC when a field is shuffled, fig 08):
`LTV` **+0.094** ≫ `Pincode` +0.035 > `PAST_LOANS_ACTIVE` +0.012 ≈ `Loan_Amount`
+0.012 > everything else (<0.005). LTV alone carries the majority of predictive
value.

**SHAP** (mean |SHAP|, figs 09–11) — top drivers:
`LTV` 0.668, `Pincode` 0.215, `past_loan_active` 0.105, `Age` 0.098,
`Product_Code_MC` 0.092, `Loan_Amount` 0.089, `down_payment_est` 0.087,
`high_ltv` 0.070, `pincode_region_5` 0.049, `ltv_band_75-85` 0.046,
`Model_Description` 0.042, `vehicle_value_est` 0.042. The SHAP summary shows the
expected monotone direction: **high LTV pushes strongly toward DECLINE; an active
prior loan pushes toward APPROVE.**

**Partial dependence** (fig 12): P(DECLINED) rises sharply with LTV above ~80%;
mild positive slope on Loan_Amount; near-flat on Net_salary and Age.

**Business translation.**
- **Collateral cushion (LTV / down-payment) is the decision** — thin equity ⇒ high
  risk. This is the field the chatbot must stress-test first.
- **Geography (target-encoded Pincode + region)** is a real secondary signal
  (regional repayment behaviour), encoded so unseen pincodes fall back to a prior.
- **Prior-loan track record** reduces risk.
- **Self-reported income is nearly ignored** by the model — the prudent behaviour
  for an NBFC facing unverified income; the model leans on collateral instead.

---

## Step 17 — Robustness & imbalance handling

**Imbalance strategy comparison** (tuned CatBoost family, 3-fold CV on train,
resampling applied *inside* folds — `reports/imbalance_study.csv`):

| Strategy | CV PR-AUC | Recall @0.5 |
|---|---|---|
| **`class_weight="balanced"` (adopted)** | **0.1589** | 0.735 |
| BalancedRandomForest | 0.1503 | 0.628 |
| SMOTE (+ no class weight) | 0.1354 | 0.035 |
| ADASYN (+ no class weight) | 0.1348 | 0.025 |

**Cost-sensitive class weighting clearly beats synthetic oversampling.** SMOTE/
ADASYN interpolate minority points in a space where the classes overlap heavily
(weak signal) — they degrade PR-AUC *and* collapse recall at 0.5. This is the
expected outcome for low-separability tabular credit data, and it is why we ship
class weighting, not resampling.

**Stability.**
- **Seed stability:** refitting with seeds {7, 21, 101} gives validation PR-AUC
  0.1665 / 0.1658 / 0.1654 → **mean 0.1659, std 0.0005** (0.3% relative). The model
  is not seed-fragile.
- **Overfitting check:** CV PR-AUC 0.159 ≈ validation 0.165 ≈ test 0.174, and
  CV/val/test ROC-AUC all ≈ 0.76–0.78. The tiny train→test gap confirms the
  regularised booster generalises rather than memorises — exactly what the brief's
  hold-out framing asks for.

---

## Step 18 — Final model selection & justification

**Chosen model: tuned CatBoost** (class-weighted), threshold 0.59.

| Criterion | Why CatBoost |
|---|---|
| Performance | Best CV PR-AUC (0.159); test ROC-AUC 0.778, PR-AUC 0.174 — top of a tight pack. |
| Generalisation | CV ≈ val ≈ test; seed-std 0.0005. No overfitting. |
| Robustness | Handles mixed types, missing values, and unseen categories gracefully; regularised. |
| Latency | 20 ms/applicant single-call — far inside any real-time SLA. |
| Interpretability | Global + local SHAP available; drivers are sensible and defensible to a regulator. |

**Honest caveat:** a plain **Logistic Regression** was within ~0.01 PR-AUC (0.148)
and is fully transparent + sub-millisecond. If the credit-risk committee prioritises
maximum explainability/auditability over the last ~7% of PR-AUC, LogReg is a
legitimate, defensible fallback — it is retained in the zoo for exactly that reason.
We ship CatBoost for the marginal performance and richer interaction capture, with
LogReg as the interpretable challenger.

## Step 19 — Persistence (saved artifacts)

`artifacts/` (everything needed for inference from **raw** applicant inputs):
- `model_pipeline.joblib` — the full FeatureEngineer → preprocessing → CatBoost
  pipeline (imputers, encoders, scaler, target-encoder all fit and frozen).
- `model_metadata.json` — family, tuned hyperparameters, **threshold 0.59**,
  threshold policy, label mapping (APPROVED=0 / DECLINED=1), the 93 feature names,
  full test metrics, imbalance study, seed stability.

Inference needs **only raw fields** — no manual preprocessing. Reproducible via a
single `RANDOM_SEED`.

## Step 20 — Prediction function

`src/predict.py :: predict_customer(raw_applicant: dict)` returns:
`prob_approve`, `prob_decline`, `decision`, `threshold`, `confidence`,
`risk_explanation` (rule-based, EDA-grounded), `missing_fields`, `model_version`.
The **decision is the model's** (`prob_decline ≥ threshold`); the explanation only
narrates it. Live examples:

| Applicant | P(decline) | Decision | Lead reason |
|---|---|---|---|
| high-risk (97% LTV, student, no income, no prior loan) | 0.772 | **DECLINED** | Very high LTV (97%) — minimal down-payment |
| low-risk (55% LTV, salaried ₹60k, prior loan, EV) | 0.025 | **APPROVED** | Healthy LTV (55%) — sizeable down-payment |
| borderline (84% LTV, self-employed ₹28k) | 0.548 | APPROVED | No prior loan on record (near the 0.59 line) |

## Step 21 — Production readiness

| Concern | Recommendation |
|---|---|
| **Logging / repository** (Objective d) | Implemented: every session appends to `logs/applications.jsonl` — inputs (PII-masked), engineered values, model score, threshold, decision, confidence, non-reversible applicant hash. |
| **PII** | Pincode masked before persistence (`5XXXXX`); no raw pincode stored. Extend masking to name/phone if collected. Encrypt at rest; restrict access. |
| **Versioning** | Version the joblib + metadata (semver + git SHA + training-data hash); log `model_version` on every decision for auditability. |
| **Data validation** | Validate ranges/enums at the API edge (Age 15–100, LTV 0–100, known enums) — the pipeline tolerates unseen categories, but reject impossible numerics early. |
| **Drift monitoring** | Track input distributions (LTV, loan, tier mix), score distribution, and approval rate weekly (PSI / KS vs training). Alert on decline-rate or feature drift. |
| **Performance monitoring** | Once repayment outcomes mature, monitor realised PR-AUC / recall / approval-rate and calibration on live decisions. |
| **Retraining** | Scheduled quarterly + drift-triggered; always re-tune threshold on fresh validation. Keep a champion/challenger (CatBoost vs LogReg) shadow test. |
| **Latency** | Measured **p99 = 21 ms** end-to-end; batch 0.008 ms/applicant. Comfortable for a synchronous chatbot call. Cache the loaded pipeline in the serving process. |
| **Fairness** | `Gender` is a weak predictor and a protected attribute — monitor approval-rate parity across gender/geography; consider dropping `Gender` (negligible Δ PR-AUC) to reduce disparate-impact risk. |

---

## Step 22 — Probability calibration & adverse-action reason codes (regulatory extension)

The class-weighted CatBoost ranks well but its raw scores are **not** probabilities
(weighting inflates them toward 0.5 — raw ECE 0.35). For regulatory use (IND-AS 109
/ ECL provisioning, risk-based pricing, adverse-action notices) we add a calibration
layer and standardized reason codes. `src/calibration.py`, `08_calibrate.py`.

**Calibration.** `IsotonicRegression` fit on the **validation** holdout (test
untouched); monotone, so ranking is preserved. Measured on TEST:

| | Raw score | **Calibrated PD** |
|---|---|---|
| Brier score | 0.196 | **0.047** |
| Expected Calibration Error | 0.346 | **0.0027** |
| Log-loss | 0.566 | **0.180** |
| Mean predicted PD vs actual bad rate | — | **5.24% vs 5.26%** |

Reliability diagram + PD distribution in **fig 13**. The score is now a genuine PD:
predicted 5.2% ≈ observed 5.3% in aggregate, and near-diagonal per decile.

**Interpretable decision rule.** The operating point becomes a PD cutoff:
**DECLINE if calibrated PD ≥ 6.9%.** Deployed calibrated model on TEST at this
cutoff: recall (DECLINED) **0.651**, precision 0.119, accuracy 0.729, ROC-AUC 0.776,
balanced-acc 0.692, MCC 0.190, log-loss **0.180**; confusion TN 11,293 · FP 4,114 ·
**FN 298 · TP 557** (catches 65% of risky applicants). *(This calibrated model is
trained on TRAIN with VAL reserved for calibration; the pre-calibration model in
§13 was refit on TRAIN+VAL — both are reported honestly.)*

**Reason codes (adverse action).** Per-applicant SHAP contributions are aggregated
from the 93 encoded features up to business concepts and ranked; the concepts
pushing hardest toward DECLINE become the reasons on the notice. Example (declined,
PD 13.1%): `LTV_LOW_EQUITY` (+1.30) → `VEHICLE_RISK` → `NO_CREDIT_HISTORY` →
`AGE_FACTOR`. Approvals emit the top *favourable* factors. Codes are logged with
every decision for audit. **Deployed artifact: `artifacts/model_calibrated.joblib`.**

---

## Step 23 — Improvement investigation (can we do better?)

A dedicated effort to beat the tuned-CatBoost baseline (5-fold CV PR-AUC 0.162).

**Diagnostics.**
- *Learning curve* (fig 14): PR-AUC 0.129 → 0.158 from 5k→76k rows; the last
  doubling adds **+0.0003** → **data-saturated**. More rows of the same fields
  won't help.
- *Feature ablation*: neutralising **LTV/collateral** drops PR-AUC by **−0.018**
  (by far the largest) — the load-bearing signal, corroborating permutation/SHAP.
- *Bayes-error proxy*: kNN neighbourhood label-disagreement ≈ 5.7% (≈ base rate),
  implied local max-AUC ≈ 0.705 — a crude, high-dimension-pessimistic estimate
  (our model already exceeds it), so not relied upon; the plateau + cross-model
  agreement are the real ceiling evidence.

**Squeeze experiments** (5-fold CV PR-AUC vs 0.1619 baseline, noise ±0.012):

| Config | PR-AUC | Δ | Verdict |
|---|---|---|---|
| baseline tuned CatBoost | 0.1619 | — | reference |
| stacking (Cat+HistGB+LogReg) | 0.1621 | +0.0002 | within noise |
| + interaction features | 0.1608 | −0.0011 | GBMs already model interactions |
| soft-voting | 0.1607 | −0.0012 | no help |
| monotonic constraints | — | — | defensibility feature, not an accuracy lever |

**Neural net (MLP)** — closing the "did you try deep learning?" question:

| Config | PR-AUC | ROC-AUC | Δ vs baseline |
|---|---|---|---|
| MLP (plain) | 0.1445 | 0.751 | −0.017 (worse than every booster) |
| MLP (oversampled) | 0.1005 | 0.678 | −0.061 (resampling wrecks it) |
| stack + MLP | 0.1642 | 0.771 | +0.0023 (best seen, but < noise) |

Standalone the MLP is clearly worse — the expected result for tabular data at this
scale. As a diverse stack member it produced the highest PR-AUC observed (0.1642),
but the gain is inside the noise band and does not justify shipping a neural net.

**Verdict: keep the tuned CatBoost.** 15 models/ensembles all plateau at ROC
~0.75–0.78. The ceiling is **information, not model capacity**.

**New-data scaffolding (Bucket B — the real lever).** `src/retrain.py` provides
plug-and-play hooks for the step-change data we don't yet have:
- `enable_bureau_score()` — a bureau-score (CIBIL/Experian) feature slot; the
  pipeline already tolerates it. Industry benchmark: a bureau score typically lifts
  application scorecards to Gini 0.55–0.70.
- `make_default_label()` + `retrain_with_default_label()` — swap the historical-
  decision proxy for a **realised-default** target (90+ DPD) and apply **reject
  inference** (fuzzy augmentation) so the model isn't biased by the accept-only
  sample. This is what turns the model from decision-replication into a true PD model.

---

## Deliverable — X (time to complete) & Y (decision latency)

**X = 3 minutes (target application completion time).** The model needs 15 raw
fields, but the chatbot asks **~10 questions**: a single vehicle pick auto-fills
`Make_Code` / `Model_Description` / `Model_Variant` / `Product_Code`, and
`Final_Tier` is derivable from the pincode. Most answers are taps/selects. At ~10
questions, ~15–18 s each including validation, a median applicant finishes in
**≈2.5–3 min** — short enough to keep abandonment low (the brief warns that too many
questions deter completion), yet enough to collect the fields that matter.

*Field-count optimisation (part of the task):* permutation importance shows
`LTV`, `Pincode`, `PAST_LOANS_ACTIVE`, `Loan_Amount` carry ~95% of the signal. A
reduced **7–8 field** flow (drop `Gender`, `Resident_Type`, `Qualifications`, and
the descriptive model text) would lose only a sliver of PR-AUC while cutting the
form further — a worthwhile A/B test for conversion vs accuracy.

**Y = 1 second (decision SLA after final input).** Measured end-to-end inference
(clean → feature-engineer → model → threshold) is **mean 19.7 ms, p99 21 ms**; batch
is 0.008 ms/applicant. A 1-second SLA leaves >40× headroom for network + logging,
so the "Approve/Decline in ≤ Y s" objective is met with large margin.

---

## Required outputs — summary

| # | Output | Value |
|---|---|---|
| 1 | Best model | **Tuned CatBoost** (class-weighted), threshold 0.59 |
| 2 | Why chosen | Best CV PR-AUC, stable, low latency, explainable; LogReg retained as transparent challenger |
| 3 | Final hyperparameters | iterations 600, lr 0.0138, depth 5, l2_leaf_reg 4.06, random_strength 1.05, bagging_temperature 0.016, auto_class_weights=Balanced |
| 4 | Cross-validation | 5-fold CV PR-AUC 0.159; seed-std 0.0005 |
| 5 | Test metrics | see table below |
| 6 | Accuracy | 0.773 |
| 7 | Precision (DECLINED) | 0.130 |
| 8 | Recall (DECLINED) | **0.580** |
| 9 | F1 (DECLINED) | 0.212 |
| 10 | ROC-AUC | **0.778** |
| 11 | PR-AUC | **0.174** (base 0.053 → 3.3×) |
| 12 | Confusion matrix | TN 12,074 · FP 3,333 · FN 359 · TP 496 |
| 13 | Threshold | **0.59** (cost-min, FA:FR = 12:1, risk-averse) |
| 14 | Top features | LTV ≫ Pincode > PAST_LOANS_ACTIVE ≈ Loan_Amount > down_payment/high_ltv/Age (SHAP + permutation agree) |
| 15 | SHAP | LTV dominates; high LTV → decline, prior loan → approve (figs 09–11) |
| 16 | Error analysis | misses are mid-LTV (75–95) applicants indistinguishable on collected fields |
| 17 | Improvements | add bureau score / verified income / delinquency history — signal, not modelling, is the ceiling |
| 18 | Latency | 19.7 ms mean, 21 ms p99 per applicant |
| 19 | Limitations | weak intrinsic signal (ROC ceiling ~0.78); low precision at useful recall; label = historical underwriting decision (inherits its biases), not realised default |
| 20 | Future work | bureau/alt-data features, probability calibration, monitoring + quarterly retrain, fairness audit, reduced-field A/B test |

## Limitations (explicit)

1. **Signal ceiling.** With the 15 given fields the best achievable ROC-AUC is
   ~0.78; no model choice overcomes it. Precision at useful recall is low
   (declining risky applicants also declines many good ones).
2. **Label semantics.** `Decision` is the *historical underwriting outcome*, not a
   realised default. The model learns to replicate past decisions and inherits any
   bias in them — it is a decision-support/consistency tool, not ground-truth risk.
3. **Self-reported, unverified income** carries little signal; verified income would
   likely help materially.
4. **Calibration** was not enforced; if probabilities feed downstream pricing,
   add isotonic/Platt calibration on a held-out fold.

*This report is regenerable from the saved artifacts + `reports/*.json`/`*.csv`.*
