# Thin pod image. Contains everything that rarely changes (Python + Node + claude CLI
# + autoresearch's pinned third-party deps), but NOT the autoresearch source itself.
# The entrypoint git-clones autoresearch fresh at every pod boot, so code changes
# ship by `git push main` + dispatching a new pod — no image rebuild needed.
#
# Image rebuild only required when: a Python dep is added/upgraded in pyproject.toml,
# an apt package is added, the RunPod base bumps, or the Node / claude CLI version
# changes. See .github/workflows/build-image.yml for the path filter that enforces
# this.
#
# MODE=serve is no longer supported on this image (Railway builds the controller
# from source via `railway up`).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# System packages + Node + claude CLI + hf-transfer. openssh-server is added
# explicitly so the entrypoint's sshd debug path works on a minimal base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl gnupg openssh-server \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir hf-transfer

# Install ONLY autoresearch's third-party deps (extracted from pyproject.toml).
# The autoresearch package itself is git-cloned at runtime by the entrypoint;
# at that point we run `pip install --no-deps -e .` against the clone, which
# only registers the package + console script (the deps are already here).
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -c "import tomllib; \
import sys; \
deps = tomllib.load(open('/tmp/pyproject.toml','rb'))['project']['dependencies']; \
print('\n'.join(deps))" > /tmp/deps.txt \
    && pip install --no-cache-dir -r /tmp/deps.txt \
    && rm /tmp/pyproject.toml /tmp/deps.txt

# The entrypoint git-clones autoresearch into /app at runtime.
RUN mkdir -p /app
WORKDIR /app

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENV MODE=pod
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
