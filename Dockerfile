FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# HF Spaces runs the container as uid 1000; anything the app writes must be
# owned by that user or startup fails with EACCES.
RUN useradd -m -u 1000 user

WORKDIR /app

# No apt layer: git is gone because nothing is installed from a git URL any
# more (the Hebrew phonikud/style-onnx git deps were dropped), and every
# remaining dependency ships wheels.

# Install into the system interpreter — a venv buys nothing in a container.
ENV UV_SYSTEM_PYTHON=1

COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# What is NOT copied is decided by .dockerignore (.venv, .git, __pycache__,
# *.wav, docs/, scripts/), and it has to be decided there. A `COPY . .` followed
# by `RUN rm -rf` shrinks nothing: layers are additive, so the copied 256 MB
# venv and 354 MB git directory would stay in the COPY layer and the removal
# would only add whiteouts. Spaces builds from git and never sees them; a laptop
# build now matches it.
COPY . .

# --- the Hugging Face cache ------------------------------------------------
# This container downloads TWO repos at runtime, ~1.5 GB together:
#   notmax123/phonikud-yi-engine   1.23 GB  (v5 pointing model + G2P tables)
#   notmax123/blue-yi              281 MB   (the default acoustic runtime)
#
# The single most common HF Spaces Docker failure: huggingface_hub defaults its
# cache to $HOME/.cache/huggingface, $HOME defaults to /root for a root-built
# image, and the process runs as uid 1000 — so snapshot_download dies part-way
# through the transfer with a permission error that reads like a network fault.
# Three things have to agree, and all three are set here: HOME, HF_HOME, and
# ownership of the directory they point at. HF_HOME is what huggingface_hub
# actually reads (HUGGINGFACE_HUB_CACHE is derived from it as $HF_HOME/hub), so
# pinning HOME alone is not enough — a library that resolves the cache before
# HOME is exported would still land in /root.
#
# Both repos share this one cache: the engine bridge honours
# PHONIKUD_YI_ENGINE_DIR and the Blue adapter honours BLUE25_MODEL_DIR, but
# neither is set here, so both fall through to snapshot_download and land in
# HF_HOME. XDG_CACHE_HOME is pinned too, because onnxruntime and matplotlib-ish
# transitive packages write there and inherit the same /root problem.
ENV HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/.cache/huggingface/hub && chown -R user:user /app
USER user

# --- prewarm: still off by default, and now for a stronger reason ----------
# With one 1.23 GB repo this was a close call. With two repos totalling ~1.5 GB
# it is not: baking them in costs ~1.5 GB of image layer AND the same ~1.5 GB
# again in the container's own writable cache the first time anything resolves
# a symlink into it, so the deployed footprint roughly doubles for a saving
# that only ever applies to the very first request after a rebuild. Worse, an
# image that contains the models has to be rebuilt and re-pushed for every
# engine or voice-bundle revision, which is exactly the coupling that keeping
# the models in their own repos was meant to remove.
#
# So the default stays: the models are fetched at runtime by app.py's startup
# hook on a background thread. The port binds immediately (Spaces kills a
# container that misses its health window), the page and /docs serve straight
# away, /health reports `warming`, and only the first synthesis request waits.
#
# Build with --build-arg PREWARM_MODELS=1 to bake them in anyway — for an
# air-gapped or offline deployment, or when instant cold starts are worth the
# image size. It runs as `user` with HF_HOME already exported, so the cache it
# writes is the same one the app reads.
ARG PREWARM_MODELS=0
RUN if [ "$PREWARM_MODELS" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download as d; \
d('notmax123/phonikud-yi-engine'); d('notmax123/blue-yi')"; \
    fi

EXPOSE 7860

# 0.0.0.0 so the Spaces proxy can reach it; no --reload (the old Flask app ran
# debug=True in the published Space, which exposes a console to the internet);
# one worker, because both the engine and the acoustic runtime are process-wide
# singletons holding hundreds of megabytes of ONNX sessions and a second worker
# would double that for no throughput gain on CPU-basic hardware.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
