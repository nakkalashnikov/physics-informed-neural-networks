# ── Base image ────────────────────────────────────────────────────────────────
# PyTorch 2.3 + CUDA 12.1 + cuDNN 8 (runtime variant — smaller than devel).
# Build on Vast.ai with an NVIDIA GPU (RTX 3090 / A100 recommended).
# To build locally on Apple Silicon for testing: use `--platform linux/amd64`
# and swap to a CPU image, since MPS does not exist inside Docker on macOS.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="pinn-training-service"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY config.yaml  .
COPY model.py     .
COPY physics.py   .
COPY sampler.py   .
COPY trainer.py   .
COPY train.py     .

# ── Checkpoint volume ─────────────────────────────────────────────────────────
# Mount a host directory here to persist checkpoints after the container exits:
#   docker run -v $(pwd)/checkpoints:/workspace/checkpoints ...
VOLUME ["/workspace/checkpoints"]

ENV PYTHONUNBUFFERED=1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "train.py", "--config", "config.yaml"]
