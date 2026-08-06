# ABC Credit — Loan Approval Prediction Model

Production-quality, leakage-safe binary classifier that predicts **APPROVED /
DECLINED** for two-wheeler loan applications, plus the chatbot interaction layer
that collects features and explains the decision. The **credit decision is made
solely by the trained ML model** against a business-chosen probability threshold;
any generative/LLM layer stays in the interaction layer (per the project brief).

## Why this problem is hard (read first)
- **Severe class imbalance:** only **5.26%** of historical applications were
  DECLINED (~18:1). Accuracy is meaningless; we rank on **PR-AUC** and tune the
  threshold for **recall on the DECLINED class** (catching risky applicants).
- **Genuinely limited signal:** the strongest predictor (LTV) has a univariate
  AUC of ~0.74; every strong model plateaus near **ROC-AUC 0.76–0.77**. We build
  for generalisation and defensible thresholds, not a vanity metric.

## Project layout
```
abc-credit-model/
├── data/                     training_data.csv, project_brief.txt
├── src/
│   ├── config.py             column roles, paths, seeds, business constants
│   ├── cleaning.py           reproducible, stateless cleaning (Step 5)
│   ├── features.py           FeatureEngineer (+ InteractionEngineer) (Step 6)
│   ├── pipeline.py           ColumnTransformer + full pipeline (Step 8)
│   ├── data.py               load + stratified train/val/test split (Step 9)
│   ├── modeling.py           model zoo, CV scorers, full metric block
│   ├── calibration.py        isotonic PD calibration + reason codes (Step 22)
│   ├── predict.py            predict_customer() inference API (Step 20)
│   ├── retrain.py            bureau-score slot + reject-inference hooks (Bucket B)
│   ├── chatbot.py            rule-based questionnaire + PII masking + logging (Deliverable 3)
│   ├── llm_client.py         Claude Haiku 4.5 wrapper + offline fallback (interaction layer)
│   └── chatbot_llm.py        LLM chatbot orchestrator — decision stays in the ML model
├── 00_profile.py … 05_explain.py   profile / EDA / selection / tune / final-eval / explain
├── 07_finalize_all.py        adopt threshold + latency + chatbot check
├── 08_calibrate.py           isotonic PD calibration + reason codes
├── 09_diagnostics.py 10_improve.py 11_squeeze2.py 12_mlp.py 13_model_table.py   (improvement study)
├── build_assets.py           vehicle catalog + pincode→tier + region→tier lookups
├── api.py                    hardened FastAPI service (auth, rate limit, size limits)
├── loadtest.py               concurrent /decision load test
├── Dockerfile / .dockerignore  container image (py3.11-slim + libgomp)
├── tests/test_security.py    boundary + auth + rate-limit + injection tests
├── artifacts/                model_calibrated.joblib (deployed), model_pipeline.joblib,
│                             model_metadata.json, vehicle_catalog / pincode_tier / region_tier.json
├── reports/                  REPORT.md, CHATBOT_PLAN.md, PRODUCTION.md, figures/, *.json, *.csv
└── logs/                     applications.jsonl (PII-masked chatbot repository)
```

## Setup (from scratch)

**Prerequisites**
- **Python 3.11**
- **OpenMP runtime** (required by CatBoost/XGBoost/LightGBM native libs):
  - macOS: `brew install libomp`
  - Debian/Ubuntu: `sudo apt-get install -y libgomp1`
- **`uv`** (fast installer) — `curl -LsSf https://astral.sh/uv/install.sh | sh` — *or* use stdlib `venv` + `pip` (shown below).

**1 — Data files.** Put the two provided files in `data/` (already present here):
`data/training_data.csv` and `data/project_brief.txt`. *(Serving only needs
`artifacts/`; the CSV is required only to re-train.)*

**2 — Environment + dependencies**
```bash
cd abc-credit-model

# Option A — uv (recommended)
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# Option B — stdlib venv + pip
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**3 — Verify the install**
```bash
.venv/bin/python -c "import numpy,pandas,sklearn,catboost,xgboost,lightgbm,shap,fastapi,anthropic; print('env OK')"
.venv/bin/python build_assets.py          # build chatbot lookups -> artifacts/
.venv/bin/python -m pytest tests/ -q      # expect: 10 passed
```
> If step 3 fails with `libxgboost.dylib … libomp.dylib not loaded` (or a
> `lib_lightgbm` OpenMP error), the OpenMP runtime above isn't installed — install
> it and re-run. CatBoost bundles its own, so the model still loads without it.

**4 — (Optional) runtime env vars**
- `ABC_API_KEY` — REST API auth (unset = dev-open mode, flagged in `/health`)
- `ANTHROPIC_API_KEY` — enables the Claude Haiku chatbot layer (unset = deterministic offline)

---

## Reproduce end-to-end
Assumes the environment from **Setup** is active.
```bash
.venv/bin/python 00_profile.py          # data quality
.venv/bin/python 01_eda.py              # EDA
.venv/bin/python 02_model_selection.py  # feature selection + 5-fold CV comparison
.venv/bin/python 03_tune.py 30          # Optuna tuning
.venv/bin/python 04_final_eval.py       # threshold + imbalance + test metrics + save
.venv/bin/python 05_explain.py          # SHAP / PDP / permutation importance
.venv/bin/python 07_finalize_all.py     # adopt threshold + latency + chatbot check
.venv/bin/python 08_calibrate.py        # isotonic PD calibration + reason codes
```

**Deployed model:** `artifacts/model_calibrated.joblib` — CatBoost (train-fit) +
isotonic PD calibration (val-fit). Emits a **calibrated Probability of Default** and
**adverse-action reason codes**. Decision rule: *DECLINE if PD ≥ 6.9%*.

## Inference
```python
from src.predict import predict_customer
predict_customer({...raw applicant fields...})
# -> {prob_approve, prob_decline, decision, threshold, confidence, risk_explanation, ...}
```

## Chatbot
Two front-ends over the same deterministic decision (`predict_customer()`):

```bash
# Rule-based questionnaire:
.venv/bin/python src/chatbot.py

# LLM chatbot (Claude Haiku 4.5 interaction layer; decision stays in the ML model).
# First build the lookup assets (one-time):
.venv/bin/python build_assets.py                # vehicle catalog + pincode→tier
.venv/bin/python src/chatbot_llm.py             # offline (no key) OR:
export ANTHROPIC_API_KEY=sk-ant-... && .venv/bin/python src/chatbot_llm.py   # live
```
The LLM only collects inputs + explains results — it never approves/declines.
Runs fully offline (deterministic) without a key. See `reports/CHATBOT_PLAN.md`.

## REST API (FastAPI)
```bash
.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
# GET  /health      service + interaction-layer status
# GET  /vehicles    popular vehicle picker list
# POST /decision    raw applicant fields -> PD + APPROVE/DECLINE + reason codes (pure ML)
# POST /normalize   free-text answer -> canonical field value (slot-filling)
# POST /application collected chatbot answers -> decision + explanation + reason codes
```
Example:
```bash
curl -s -X POST localhost:8000/decision -H 'Content-Type: application/json' \
  -d '{"applicant":{"Loan_Amount":135000,"LTV":97.0,"Age":20,"Employment_Type":"STU",
       "Pincode":"509210","PAST_LOANS_ACTIVE":"NO_PAST_LOANS","Net_salary":0,
       "Product_Code":"MC","Make_Code":"RAIDER","Model_Description":"TVS RAIDER DISC BSVI",
       "Model_Variant":"RAIDER","Final_Tier":"07 Rural","Resident_Type":"R","Gender":"Male",
       "Qualifications":"SSC"}}'
# -> {"decision":"DECLINED","calibrated_pd":0.1309,"reason_codes":[...]}
```
The decision is always computed by the ML model. Set `ANTHROPIC_API_KEY` to switch
the interaction layer to Claude Haiku 4.5 (`/health` reports which is active).
Security: `X-API-Key` auth (env `ABC_API_KEY`), per-client rate limiting, request-size
+ field limits. See `reports/PRODUCTION.md`.

## Tests + deploy
```bash
# security / boundary suite (proves a compromised LLM can't flip a decision):
.venv/bin/python -m pytest tests/ -q

# production run — multiple workers (each ≈26 req/s; ~39 ms/decision single-threaded):
ABC_API_KEY=secret RATE_LIMIT_PER_MIN=120 \
  .venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# container:
docker build -t abc-credit-api .
docker run -p 8000:8000 -e ABC_API_KEY=secret -e ANTHROPIC_API_KEY=sk-ant-... abc-credit-api

# load test:
.venv/bin/python loadtest.py --url http://127.0.0.1:8000 --n 500 --concurrency 20 --api-key secret
```
Env vars: `ABC_API_KEY` (API auth), `ANTHROPIC_API_KEY` (enables Claude Haiku layer),
`RATE_LIMIT_PER_MIN`, `MAX_BODY_BYTES`. Full production/monitoring/retraining/scaling
notes: `reports/PRODUCTION.md`.

Reproducibility: single `RANDOM_SEED` (42) throughout; every transform is fit on
training folds only; unseen categories never break the pipeline (OHE
`infrequent_if_exist`, ordinal sentinel, target-encoder prior fallback).

See **reports/REPORT.md** for the full write-up (metrics, threshold justification,
SHAP, error analysis, limitations, latency, X/Y choices).
