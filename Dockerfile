FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HELMET_DEVICE=cpu \
    HELMET_MODEL_REGISTRY=/app/configs/model_registry.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir ".[service]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "helmet_safety.service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

