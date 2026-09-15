# Tier-1 GPU image for RSI-Plateau.
# Base: CUDA 12.1 + Python 3.11. Works on RunPod / Vast.ai / Lambda.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.hf_home

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rsi_plateau

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Install full GPU pipeline.
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -e ".[tier1]"

CMD ["bash"]