# ABC Credit Chatbot — Build Plan & Status

LLM chat front-end over the deployed calibrated ML model. **The ML model makes the
credit decision; the LLM only collects inputs and explains the result** (the brief's
predictive/generative separation).

## Architecture (the hard boundary)
```
applicant ─▶ Claude Haiku 4.5 (normalise each answer)     ← interaction layer, no decisions
               │
   orchestrator owns the flow (plain Python)              ← control flow = guardrail
               ▼
     derive fields (vehicle→make/model/product, price+loan→LTV, pincode→tier)
               ▼
     predict_customer()  → calibrated PD, APPROVE/DECLINE@6.9%, reason codes  ← DECISION (ML)
               ▼
     Claude Haiku 4.5 (explain decision from reason codes)  ← communication layer, no decisions
               ▼
     present + PII-masked repository log
```

## Guardrails enforcing the boundary
1. **Decision computed in Python** after slots are filled — the LLM outputs *fields*, never a verdict.
2. **System prompts** (`llm_client.SLOT_SYSTEM`, `EXPLAIN_SYSTEM`): the LLM must not decide, and must use *only* the provided reason codes (no invented rationale → adverse-action safe).
3. **Orchestrator owns control flow** — the LLM never sees the flow logic or the threshold.
4. **Edge validation** (`chatbot._validate`) runs on every field regardless of LLM output.
5. **Determinism** — same inputs → same decision (ML is deterministic); the LLM affects phrasing only.

## Components
| File | Role | Status |
|---|---|---|
| `src/llm_client.py` | Claude Haiku 4.5 wrapper (normalise answer, explain) + **offline fallback** | ✅ built |
| `src/chatbot_llm.py` | Orchestrator: flow, derivation, decision, explanation, logging | ✅ built |
| `build_assets.py` → `artifacts/vehicle_catalog.json`, `pincode_tier.json` | data-derived lookups (1 vehicle pick → 4 fields; pincode → tier) | ✅ built |
| `src/predict.py` | `predict_customer()` — the decision + calibrated PD + reason codes | ✅ reused |
| `src/chatbot.py` | validators, PII masking, JSONL repository log | ✅ reused |

## Model & parameters (interaction layer)
- **Claude Haiku 4.5** (`claude-haiku-4-5`) — cheap, fast, reliable structured output; the decision isn't the LLM's job so a small model suffices.
- `max_tokens` set per call; **no temperature/top_p tuning** — reliability comes from **structured outputs** (JSON-schema slot extraction) + the system-prompt boundary, not sampling knobs.
- For data-residency (RBI), run Claude via **Bedrock/Vertex in an India region** or Claude Platform on AWS — same code, in-region.

## Applicant is asked ~9 questions (model receives 15 fields)
Vehicle (→ make/model/variant/product), vehicle price + loan (→ LTV, down-payment), pincode (→ tier),
employment, income, prior-loan, age; gender/qualification/residence optional. See `reports/REPORT.md` §fields.

## Phases
- **P1 — decision service:** `predict_customer()` wrapped. ✅
- **P2 — slot-filling:** per-answer Claude normalisation → validated schema; offline fallback. ✅
- **P3 — explanation:** Claude renders decision + reason codes; offline template fallback. ✅
- **P4 — orchestrator + logging + PII + derivation.** ✅
- **P5 — hardening (next):** REST endpoint, rate limiting, prompt-injection tests, transcript evals, pincode→tier coverage for unseen pincodes (region-digit fallback).

## Run it
```bash
# Offline (no key) — deterministic normalisation + template explanation:
.venv/bin/python src/chatbot_llm.py                 # interactive
.venv/bin/python src/chatbot_llm.py applicant.json  # scripted

# Live (Claude Haiku interaction layer):
export ANTHROPIC_API_KEY=sk-ant-...                 # or run via Bedrock/Vertex in-region
.venv/bin/python src/chatbot_llm.py
```
Verified offline end-to-end: high-risk → DECLINED (PD 13%), low-risk → APPROVED (PD ~0%),
LTV/tier/vehicle derived, reason codes direction-correct, pincode masked in the log.
```
```
