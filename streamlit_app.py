"""
ABC Credit — freeform conversational loan assistant (Streamlit).

A genuinely open-ended chat, not a fixed Q&A script: the applicant can say
anything, in any order; a Python state machine (src/chat_agent.py) tracks
which fields are still missing and keeps the conversation going via Claude
Haiku 4.5 until every required field is validated, then scores the
application with the trained ML model and reports only the verdict + next
steps (no reason codes / explanation by design).

Run:  .venv/Scripts/python.exe -m streamlit run streamlit_app.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env (if present) without adding a python-dotenv dependency, so
# ANTHROPIC_API_KEY is picked up automatically when running `streamlit run`.
_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import streamlit as st

# On Streamlit Community Cloud, secrets are configured in the app dashboard
# (Settings -> Secrets, TOML format) rather than a committed .env. Mirror them
# into os.environ so llm_client.py's plain os.environ.get() calls see them the
# same way they would locally -- regardless of the exact platform's own
# env-var-sync behaviour, this makes it explicit and dependable.
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass  # no secrets.toml / dashboard secrets configured -- fine locally

import chat_agent as agent
import llm_client

st.set_page_config(page_title="ABC Credit — Loan Assistant", page_icon="🏍️", layout="centered")

GREETING = (
    "Hi! I'm Ana from ABC Credit. 👋 Are you looking to apply for a two-wheeler "
    "loan today, or is there something else I can help with?"
)

if "display" not in st.session_state:
    st.session_state.display = [{"role": "assistant", "content": GREETING}]
if "agent_state" not in st.session_state:
    st.session_state.agent_state = agent.new_session()


def _reset():
    st.session_state.display = [{"role": "assistant", "content": GREETING}]
    st.session_state.agent_state = agent.new_session()


with st.sidebar:
    st.markdown("### ABC Credit Assistant")
    online = llm_client.online()
    st.markdown(f"**Interaction layer:** {'Claude Haiku 4.5' if online else 'offline (no API key)'}")
    st.caption("Decision is always computed by the trained ML model — the assistant only "
               "collects and validates your details; it never decides the outcome itself.")
    if not online:
        st.warning("ANTHROPIC_API_KEY not set — running in a limited offline fallback mode.")
    st.button("Start a new application", on_click=_reset, use_container_width=True)

st.title("🏍️ ABC Credit — Loan Assistant")

for msg in st.session_state.display:
    with st.chat_message(msg["role"], avatar="🏍️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

state = st.session_state.agent_state
if state.get("done") and state.get("result"):
    decision = state["result"]["decision"]
    if decision == "APPROVED":
        st.success(f"### ✅ {decision}")
    else:
        st.error(f"### ❌ {decision}")
    st.button("Start a new application", on_click=_reset, use_container_width=True, key="reset_bottom")

user_text = st.chat_input(
    "Message Ana…",
    max_chars=agent.MAX_MSG_CHARS,
    disabled=state.get("done", False),
)

if user_text:
    st.session_state.display.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant", avatar="🏍️"):
        with st.spinner("…"):
            reply = agent.respond(state, user_text)
        st.markdown(reply)
    st.session_state.display.append({"role": "assistant", "content": reply})
    if state.get("done"):
        st.rerun()
