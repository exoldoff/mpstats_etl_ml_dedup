ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.hf_cache \
    HF_HUB_CACHE=/workspace/.hf_cache/hub \
    TRANSFORMERS_CACHE=/workspace/.hf_cache/transformers

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION}

COPY pyproject.toml requirements-research.txt ./
RUN python3 -m pip install \
        "duckdb>=1.0" \
        "fastapi>=0.115" \
        "numpy>=1.24" \
        "openpyxl>=3.1" \
        "pandas>=2.0" \
        "requests>=2.31" \
        "uvicorn[standard]>=0.30"

RUN python3 -m pip install -r requirements-research.txt

COPY research ./research
COPY docs/DEDUP_FINE_TUNING_EXPERIMENT_PLAN.md docs/DEDUP_TRAINING_RUNBOOK.md ./docs/
COPY docker/dedup-training/entrypoint.sh /usr/local/bin/dedup-training

RUN chmod +x /usr/local/bin/dedup-training \
    && mkdir -p /workspace/research/dedup/data /workspace/artifacts /workspace/.hf_cache

ENTRYPOINT ["dedup-training"]
CMD ["doctor"]
