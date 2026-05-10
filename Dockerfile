FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ai.py config.yml ./

ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV HF_HOME=/tmp/.cache/huggingface

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 main:app"]
