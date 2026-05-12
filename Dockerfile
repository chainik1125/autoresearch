# Single image, two roles. MODE=serve -> controller (Railway). MODE=pod -> runner (RunPod).
#
# Base: RunPod's PyTorch 2.4 / CUDA 12.4 / Python 3.11 image. Already has torch, CUDA,
# transformers-friendly stack — saves first-pod cold-start time. ~10GB.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# git is needed for the pod entrypoint's optional `PROJECT_REPO_URL` clone.
# hf-transfer accelerates HF Hub downloads when HF_HUB_ENABLE_HF_TRANSFER=1.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir hf-transfer

# Bake the package into the image. Any package change requires rebuild + push.
# Also copy autoresearch.toml so the controller has its project config at /app —
# pydantic-settings reads from cwd, and the entrypoint runs from /app.
WORKDIR /app
COPY pyproject.toml README.md autoresearch.toml ./
COPY autoresearch ./autoresearch
RUN pip install --no-cache-dir .

# Entrypoint dispatches on $MODE.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Controller HTTP port (FastAPI). Pods don't need this published.
EXPOSE 8000

# Default to pod mode; Railway overrides with MODE=serve.
ENV MODE=pod
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
