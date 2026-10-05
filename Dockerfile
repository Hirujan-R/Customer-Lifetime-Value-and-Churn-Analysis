# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_LINK_MODE=copy

# Install uv (pinned to a major version).
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the project and install it.
COPY . .
RUN uv sync --frozen --no-dev

# Kedro stores local config and data outside the image by default.
VOLUME ["/app/data", "/app/conf/local"]

# Override at runtime, e.g.:
#   docker run --rm <image> kedro run --pipeline data_ingestion
#   docker run --rm -p 8501:8501 <image> streamlit run src/customer_clv_churn/dashboard/app.py
CMD ["kedro", "run"]
