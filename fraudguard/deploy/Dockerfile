# syntax=docker/dockerfile:1.7
# Multi-stage: build tooling never reaches the runtime image.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user with no shell and no home directory to write into.
RUN groupadd --gid 10001 fraudguard \
    && useradd --uid 10001 --gid fraudguard --no-create-home --shell /usr/sbin/nologin fraudguard

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=root:root src/ ./src/
COPY --chown=root:root artifacts/model.joblib ./artifacts/model.joblib

ENV PYTHONPATH=/app/src

# Application code is read-only to the runtime user: a compromised process
# cannot rewrite its own source or swap the model artifact.
RUN chmod -R a-w /app && mkdir -p /var/audit && chown fraudguard:fraudguard /var/audit

USER fraudguard:fraudguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

# No --reload, no debug. Workers tuned per CPU at deploy time.
CMD ["uvicorn", "fraudguard.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--no-server-header", "--proxy-headers", \
     "--forwarded-allow-ips", "10.0.0.0/8"]
