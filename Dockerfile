# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS web-build

WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS app

ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    KMP_DUPLICATE_LIB_OK=TRUE \
    MPSTATS_APP_HOST=0.0.0.0 \
    MPSTATS_APP_PORT=8055 \
    MPSTATS_WORKDIR=/app/runtime/pipeline \
    MPSTATS_DB_PATH=/app/runtime/mpstats.duckdb \
    MPSTATS_CONFIG_PATH=/app/runtime/pipeline/step1_export_config.json \
    MPSTATS_RULES_PATH=/app/runtime/classifiers/rules.csv \
    MPSTATS_MANUAL_OVERRIDES_PATH=/app/runtime/classifiers/manual_overrides.csv \
    MPSTATS_CATEGORY_CATALOG_PATH=/app/runtime/catalog/category_catalog.csv \
    MPSTATS_STATIC_DIR=/app/web/dist \
    DUCKDB_TEMP_DIRECTORY=/app/runtime/duckdb_tmp \
    HF_HOME=/app/runtime/huggingface

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libgomp1 \
    && python -m pip install --upgrade pip \
    && python -m pip install --index-url "$TORCH_CPU_INDEX_URL" "torch>=2.3,<3" \
    && python -m pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY --from=web-build /app/web/dist ./web/dist
COPY scripts/docker_entrypoint.sh /usr/local/bin/mpstats-docker-entrypoint
RUN chmod +x /usr/local/bin/mpstats-docker-entrypoint

EXPOSE 8055
ENTRYPOINT ["mpstats-docker-entrypoint"]
