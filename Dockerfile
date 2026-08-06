# ABC Credit Loan API — production image
FROM python:3.11-slim

# OpenMP runtime for CatBoost / XGBoost / LightGBM native libs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]"

# app code + serving assets (NOT the training CSV — see .dockerignore)
COPY src/ ./src/
COPY api.py build_assets.py ./
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Set ABC_API_KEY (auth) and optionally ANTHROPIC_API_KEY (LLM layer) at runtime, e.g.
#   docker run -e ABC_API_KEY=... -e ANTHROPIC_API_KEY=... -p 8000:8000 abc-credit-api
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
