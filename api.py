"""
ABC Credit — hardened REST API over the deployed decision + LLM chatbot.

Endpoints
  GET  /health         service + interaction-layer status (public)
  GET  /vehicles       popular vehicle picker list (public)
  POST /decision       raw applicant fields -> calibrated PD + APPROVE/DECLINE + reason codes
  POST /normalize      free-text answer -> canonical field value (slot-filling)
  POST /application    collected chatbot answers -> decision + explanation + reason codes

Security (Phase 5.1)
  * API-key auth  : protected endpoints require header  X-API-Key: <key>  matching
                    env ABC_API_KEY. If ABC_API_KEY is unset the API runs in
                    DEV-OPEN mode (allowed, flagged loudly in /health) — never do that in prod.
  * Rate limiting : in-memory fixed-window per client (key or IP); RATE_LIMIT_PER_MIN
                    (default 60). Returns 429 + Retry-After. (Use Redis for multi-instance.)
  * Input limits  : request body capped (MAX_BODY_BYTES, default 64 KiB -> 413) and
                    Pydantic field constraints on every payload.

The credit decision is always computed by predict_customer() (the ML model); the
LLM only normalises inputs and explains results. Set ANTHROPIC_API_KEY to enable
the Claude Haiku 4.5 interaction layer.

Run:  ABC_API_KEY=secret .venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os, sys, json, time, threading
sys.path.insert(0, "src")

from fastapi import FastAPI, HTTPException, Header, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import config as C
from predict import predict_customer
from chatbot import _validate
import chatbot_llm
import llm_client

# --------------------------------------------------------------------------- #
API_KEY = os.environ.get("ABC_API_KEY")                 # unset -> dev-open mode
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(64 * 1024)))

app = FastAPI(title="ABC Credit Loan API", version="1.1.0")


# ---- request-size guard (middleware) -------------------------------------- #
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"})
    return await call_next(request)


# ---- auth dependency ------------------------------------------------------- #
def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY is None:
        return "dev-open"                                # no key configured -> allow (dev only)
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key")
    return x_api_key


# ---- in-memory fixed-window rate limiter ---------------------------------- #
class _RateLimiter:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def check(self, client: str):
        now = time.monotonic()
        with self.lock:
            window = [t for t in self.hits.get(client, []) if now - t < 60.0]
            if len(window) >= self.per_min:
                retry = 60.0 - (now - window[0])
                raise HTTPException(status_code=429, detail="Rate limit exceeded",
                                    headers={"Retry-After": str(max(1, int(retry)))})
            window.append(now)
            self.hits[client] = window


_limiter = _RateLimiter(RATE_LIMIT_PER_MIN)


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)):
    client = x_api_key or (request.client.host if request.client else "anonymous")
    _limiter.check(client)


PROTECTED = [Depends(require_api_key), Depends(rate_limit)]


# --------------------------------------------------------------------------- #
class Applicant(BaseModel):
    applicant: dict = Field(..., description="raw applicant fields (subset of the 15 predictors)")

    @field_validator("applicant")
    @classmethod
    def _bounded(cls, v):
        if len(v) > 30:
            raise ValueError("too many fields (max 30)")
        for k, val in v.items():
            if isinstance(val, str) and len(val) > 256:
                raise ValueError(f"field '{k}' value too long (max 256 chars)")
        return v


class Answers(BaseModel):
    answers: dict = Field(..., description="chatbot answers keyed by step")
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("answers")
    @classmethod
    def _bounded(cls, v):
        if len(v) > 30:
            raise ValueError("too many answers (max 30)")
        for k, val in v.items():
            if isinstance(val, str) and len(val) > 256:
                raise ValueError(f"answer '{k}' too long (max 256 chars)")
        return v


class NormalizeReq(BaseModel):
    field: str = Field(..., max_length=64)
    kind: str = Field(default="text", max_length=16)
    choices: list | None = None
    prompt: str = Field(default="", max_length=512)
    text: str = Field(..., max_length=512)


# ---- public endpoints ------------------------------------------------------ #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "calibrated CatBoost",
        "decision_rule": "DECLINE if calibrated PD >= 6.9%",
        "interaction_layer": "claude-haiku-4-5" if llm_client.online() else "offline-deterministic",
        "auth": "enforced" if API_KEY else "DEV-OPEN (set ABC_API_KEY for production)",
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }


@app.get("/vehicles")
def vehicles():
    veh = json.loads((C.ARTIFACT_DIR / "vehicle_catalog.json").read_text())
    return {"popular": veh["popular"], "count": len(veh["catalog"])}


# ---- protected endpoints --------------------------------------------------- #
@app.post("/decision", dependencies=PROTECTED)
def decision(req: Applicant):
    """Pure ML decision service — decision made solely by predict_customer()."""
    try:
        return predict_customer(req.applicant)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/normalize", dependencies=PROTECTED)
def normalize(req: NormalizeReq):
    norm = llm_client.normalize_answer(req.field, req.kind, req.choices, req.prompt, req.text)
    value, error = _validate(req.field, req.kind, req.choices, norm if norm is not None else "")
    return {"field": req.field, "value": value, "error": error,
            "mode": "claude-haiku-4-5" if llm_client.online() else "offline-deterministic"}


@app.post("/application", dependencies=PROTECTED)
def application(req: Answers):
    try:
        return chatbot_llm.run_session(req.answers, interactive=False, session_id=req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/")
def root():
    return {"service": "ABC Credit Loan API",
            "endpoints": ["/health", "/vehicles", "/decision", "/normalize", "/application"]}
