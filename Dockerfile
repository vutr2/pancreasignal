# PancreaSignal API (Phase 6). Export the model first, then build:
#
#   python scripts/export_model.py
#   docker build -t pancreasignal-api .
#   docker run --rm -p 8000:8000 pancreasignal-api
#   curl http://127.0.0.1:8000/health
#
# Python 3.14 because requirements-lock.txt was generated with the project's Python 3.14 venv.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PANCREASIGNAL_MODEL=/app/models/urine_v1.joblib

WORKDIR /app

# Step 1: install the package + the API extra, pinned to the same versions as the lock file
COPY pyproject.toml README.md requirements-lock.txt ./
COPY src ./src
RUN pip install -c requirements-lock.txt ".[api]"

# Step 2: the model bundle and its SHA-256 file (the app refuses a bundle whose hash does not match)
COPY models/urine_v1.joblib models/urine_v1.meta.json ./models/

# Step 3: never run a web server as root
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "pancreasignal.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
