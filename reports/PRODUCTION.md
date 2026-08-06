# ABC Credit — Production & Operations Guide (Phase 5)

Covers deploying the API, and the monitoring / drift / retraining program.

## Deploy
```bash
# build
docker build -t abc-credit-api .

# run (set auth + optional LLM layer as secrets, never bake into the image)
docker run -p 8000:8000 \
  -e ABC_API_KEY="$ABC_API_KEY" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e RATE_LIMIT_PER_MIN=120 \
  abc-credit-api
```
- **Non-root** container, `HEALTHCHECK` on `/health`.
- The training CSV is **excluded** from the image (`.dockerignore`); only `artifacts/` ships.
- **Secrets** (`ABC_API_KEY`, `ANTHROPIC_API_KEY`) come from your secrets manager / orchestrator env — never committed.
- **Data residency (RBI):** to keep applicant data in-region, run the LLM layer via Claude on **Bedrock/Vertex in an India region** (or Claude Platform on AWS); the decision service is fully local either way.

## Latency & throughput (measured)
- **Per decision: ~39 ms** CPU single-threaded (model + calibrated PD + SHAP reason codes).
- **One uvicorn process ≈ 26 req/s** (CPU-bound; the GIL serialises requests). Load test:
  500 requests @ concurrency 20 → 0 errors, 26 req/s, latency p50 726 ms / p95 1133 ms — i.e.
  concurrency **queues** on a single process and inflates tail latency past the 1 s SLA.
- **Fix = process/replica parallelism**, not code: run `uvicorn --workers N` (or gunicorn +
  uvicorn workers) and/or horizontal replicas. Each worker adds ~26 req/s and keeps per-request
  latency ~40 ms when not over-subscribed. Size workers ≈ CPU cores; scale replicas to peak RPS.
- **Optional latency lever:** reason codes (the SHAP call) are the bulk of the 39 ms — for a
  high-throughput `/decision` path, compute them lazily / only when a decision is DECLINED
  (adverse-action needs them; approvals often don't).
- Load-test tool: `loadtest.py` (concurrent `/decision` → throughput + p50/p95/p99).

## Scaling notes
- The in-process rate limiter is per-instance — for multiple replicas move it to **Redis** (or an API-gateway limiter).
- Model + assets load once at process start. Scale horizontally behind a load balancer.

## Monitoring the model (once live)
| Signal | What to watch | Trigger |
|---|---|---|
| **Input drift** | PSI / KS on LTV, Loan_Amount, tier mix, employment mix vs the training distribution | PSI > 0.2 on any key feature |
| **Score drift** | mean predicted PD, PD distribution shape | sustained shift vs training mean (5.2%) |
| **Approval-rate drift** | daily approve/decline rate | large deviation from expected |
| **Calibration** | predicted PD vs realised bad rate (once repayment outcomes mature) | reliability curve departs the diagonal / Brier degrades |
| **Realised performance** | recall / precision / PR-AUC on matured loans | drop below the launch baseline (PR-AUC 0.17) |
| **Fairness** | approval-rate parity across gender / region | disparate-impact ratio outside policy band |
| **Latency** | p95 / p99 of `/decision` and `/application` | p99 > SLA (Y = 1 s; measured ~20 ms) |
| **LLM layer** | Claude error/timeout rate, cost/session | error spike or cost anomaly |

## Retraining
- **Cadence:** quarterly + drift-triggered. Always re-tune the threshold on fresh validation and **re-fit calibration** on a held-out slice.
- **Champion/challenger:** shadow-score CatBoost vs LogReg (interpretable challenger) on live traffic before promoting.
- **Target upgrade (the real lever):** move from the historical-decision label to a **realised-default** label (90+ DPD) with **reject inference**, and add a **bureau score** feature — `src/retrain.py` (`make_default_label`, `retrain_with_default_label`, `enable_bureau_score`) has the plug-and-play hooks.

## Audit & governance
- Every decision logs (PII-masked) inputs, calibrated PD, threshold, decision, reason codes, model version → `logs/applications.jsonl` (ship to your log store).
- Version the model artifact + metadata (semver + git SHA + training-data hash); log `model_version` on every decision.
- **Boundary is enforced and tested:** the LLM never decides — `tests/test_security.py` proves a compromised LLM (adversarial slot-filling + explanation) cannot flip the outcome.

## Security posture (implemented)
- API-key auth (`X-API-Key` vs `ABC_API_KEY`); dev-open mode is flagged loudly in `/health`.
- Per-client fixed-window rate limiting (`RATE_LIMIT_PER_MIN`).
- Request-body size cap (`MAX_BODY_BYTES`) + Pydantic field constraints.
- PII (pincode) masked before logging.
