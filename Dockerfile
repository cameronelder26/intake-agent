FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code edits don't bust the layer.
COPY pyproject.toml README.md LICENSE ./
COPY intake_agent ./intake_agent
RUN pip install ".[pg]"

COPY data ./data
COPY evals ./evals

# Run as a normal user; state (queue, audit log) goes to a volume.
RUN useradd --create-home intake && mkdir -p /app/.intake && chown -R intake /app
USER intake
ENV INTAKE_STATE_DIR=/app/.intake

EXPOSE 8000

# CLI by default. The API runs with:
#   docker run --entrypoint uvicorn -p 127.0.0.1:8000:8000 intake-agent \
#       intake_agent.api:app --host 0.0.0.0 --port 8000
ENTRYPOINT ["intake"]
CMD ["--help"]
