FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY hycoder/ hycoder/
RUN pip install --no-cache-dir -e .

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

ENV HYBRID_PORT=5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:${HYBRID_PORT}/api/system/version || exit 1

CMD ["sh", "-c", "ai-plus web --port $HYBRID_PORT"]
