# GCP Cloud Run deployment — CUDA-enabled for GPU inference (NVIDIA L4)
# Build: docker build -t osllm-backend .
# Run locally: docker run -e PORT=8080 -e HF_TOKEN=<token> -p 8080:8080 osllm-backend

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py ai.py config.yml ./

# HuggingFace model cache lives here; Cloud Run mounts are ephemeral,
# but with --min-instances=1 the container stays warm across requests.
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Pre-bake the model into the image to eliminate runtime downloads
# and avoid Cloud Run tmpfs OOM during cold start.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from transformers import AutoProcessor, AutoModelForCausalLM; import os; token = os.environ.get('HF_TOKEN'); AutoProcessor.from_pretrained('google/gemma-4-E2B-it', token=token); AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B-it', token=token, dtype='auto', device_map='cpu')"

# Cloud Run sets PORT; gunicorn binds to it at startup.
# --workers 1: each worker loads the full model into VRAM — one is enough.
# --timeout 600: first request may trigger a ~10GB model download.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 main:app"]
