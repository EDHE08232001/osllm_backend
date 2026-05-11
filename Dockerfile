FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ai.py config.yml ./

ENV HF_HOME=/opt/hf-cache
ARG HF_TOKEN
ARG MODEL_ID=google/gemma-4-E2B-it
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', token='${HF_TOKEN}')"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 main:app"]
