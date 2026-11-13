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

ENTRYPOINT ["intake"]
CMD ["--help"]
