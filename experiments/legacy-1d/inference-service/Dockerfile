# ── Base image ────────────────────────────────────────────────────────────────
# Same base as training-service: includes PyTorch + CUDA.
# On Apple Silicon the container runs CPU-only (MPS is unavailable inside Docker);
# the predictor auto-detects and falls back to CPU gracefully.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="pinn-inference-service"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY model.py      .
COPY predictor.py  .
COPY schemas.py    .
COPY main.py       .

# ── Checkpoint volume ─────────────────────────────────────────────────────────
# Mount the trained model here:
#   docker run -v /path/to/checkpoints:/workspace/checkpoints ...
# Or override the path via environment variable:
#   docker run -e MODEL_PATH=/workspace/checkpoints/model_10000.pt ...
VOLUME ["/workspace/checkpoints"]

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/workspace/checkpoints/model_final.pt

EXPOSE 8000

# ── Default command ───────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
