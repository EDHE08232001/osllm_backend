# Cloud Deployment Guide

Step-by-step instructions to deploy this Flask + Gemma 4 backend to **Google Cloud Run** as cheaply as possible (scale-to-zero, pay-per-request, CPU-only).

> **Scope.** This guide covers GCP Cloud Run only. AWS App Runner (12 GB RAM ceiling) and Azure Container Apps consumption (8 GB ceiling) don't fit Gemma 4 E2B at bf16 without quantization, so they're omitted. See the bottom of this doc for upgrade paths.

---

## What this deploys

- The repo **as-is** — `main.py`, `ai.py`, `config.yml`, and a modified `Dockerfile` that bakes the model into the image
- Target service: **Cloud Run** with `min-instances=0` (scale-to-zero) and CPU-only inference
- Cost expectation: **under $5/month** for hobby use, ~$0 when idle, ~$1.50/month image storage floor

Known limitations of the code (in-memory sessions, no auth, single worker) are listed at the end. Fix those before opening to real traffic.

---

## Prerequisites

1. **Hugging Face account** with the [Gemma 4 license](https://huggingface.co/google/gemma-4-E2B-it) accepted, and a token with read access — save as `HF_TOKEN`.
2. **Docker** installed locally — `docker --version` should work.
3. **gcloud CLI** installed and logged in — `gcloud --version` should work.
4. A **GCP project with billing enabled**. New accounts get $300 of free credit.

Set up shell variables you'll reuse throughout this guide:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO="osllm"
export SERVICE="osllm-backend"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

gcloud config set project "$PROJECT_ID"
```

---

## Step 1: Bake the model into the Docker image

When you deployed earlier, the model was downloaded from Hugging Face on first request and lived in `/tmp`. On Cloud Run, `/tmp` is wiped when the instance scales down — so the next cold start re-downloaded ~10 GB of weights. Baking the model into the image puts the weights in a persistent image layer so they're already there when the container starts.

**Replace your current `Dockerfile` with this:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ai.py config.yml ./

# Bake model weights into the image so Cloud Run cold starts don't
# re-download from Hugging Face. HF_HOME points at an image-layer path,
# NOT /tmp (which is wiped between instances).
ENV HF_HOME=/opt/hf-cache
ARG HF_TOKEN
ARG MODEL_ID=google/gemma-4-E2B-it
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='${MODEL_ID}', token='${HF_TOKEN}')"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 main:app"]
```

Note: the deprecated `TRANSFORMERS_CACHE` env var is removed; only `HF_HOME` is needed.

**Optional but recommended:** in `config.yml`, set `cache.preload_model: true` so the model loads during container startup (with `--cpu-boost`, below) instead of on the first request.

**Build the image locally:**

```bash
docker build \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  -t osllm-backend:latest \
  .
```

This will take a few minutes the first time and produce an image around 10–15 GB.

**(Optional) quick local smoke test:**

```bash
docker run --rm -p 8080:8080 -e HF_TOKEN="$HF_TOKEN" osllm-backend:latest
# in another terminal:
curl http://localhost:8080/health
```

---

## Step 2: One-time GCP setup

Enable the APIs you need, create the registry, and store the HF token in Secret Manager.

```bash
# Enable required services
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com

# Create the Artifact Registry repository
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Open Source LLM backend"

# Store the HF token as a secret
printf '%s' "$HF_TOKEN" | gcloud secrets create hf-token --data-file=-
# (If you ever need to rotate: replace `create` with `versions add`)

# Grant Cloud Run's default runtime SA access to that secret
export PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Step 3: Manual deploy (first time)

Push the image to Artifact Registry and create the Cloud Run service.

```bash
# Let docker push to AR
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Tag and push
docker tag osllm-backend:latest "$IMAGE"
docker push "$IMAGE"
```

Deploy:

```bash
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --cpu=8 \
  --memory=32Gi \
  --cpu-boost \
  --min-instances=0 \
  --max-instances=2 \
  --concurrency=1 \
  --timeout=600 \
  --port=8080 \
  --allow-unauthenticated \
  --set-secrets=HF_TOKEN=hf-token:latest
```

**Flag rationale:**

| Flag | Why |
|---|---|
| `--cpu=8 --memory=32Gi` | Gemma 4 E2B at bf16 needs ~10 GB just for weights; add activations + Python overhead and you want headroom. 8/32 is the max Cloud Run tier and the safest "make it work" setting. You can try 6/24 later to save a bit. |
| `--cpu-boost` | Gives more CPU during startup so the model loads quickly during the startup probe window |
| `--min-instances=0` | Scale to zero when idle (the whole point of the cheapest tier) |
| `--max-instances=2` | Cap blast radius if something goes viral |
| `--concurrency=1` | The app uses `gunicorn --workers 1` (one model in memory, not thread-safe for parallel inference). One request per instance keeps it honest. |
| `--timeout=600` | Cold start + first inference can take 30–60s on CPU |
| `--allow-unauthenticated` | Lets you `curl` it for testing. **Remove this** before exposing publicly — there's no auth in the app. |
| `--set-secrets=HF_TOKEN=hf-token:latest` | Inject the HF token from Secret Manager at runtime (the code reads `os.getenv("HF_TOKEN")` in `ai.py:66`) |

Get the URL and smoke-test:

```bash
export SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')
echo "$SERVICE_URL"

curl "$SERVICE_URL/health"
# expect: {"status": "healthy", ...}

curl -X POST "$SERVICE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, who are you?"}'
# first request after a cold start will take ~30–60s; subsequent requests faster while the instance is warm
```

---

## Step 4: CI/CD with GitHub Actions

For ongoing deploys you don't want to `docker build && push` from your laptop. The cleanest path is GitHub Actions + Workload Identity Federation (no long-lived service-account JSON keys).

### One-time WIF setup (run this once locally)

```bash
# Service account that GH Actions will impersonate
gcloud iam service-accounts create gh-actions-deployer \
  --display-name="GitHub Actions Deployer"

export GH_SA="gh-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

# Permissions: push to AR, deploy to Cloud Run, act as runtime SA
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${GH_SA}" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${GH_SA}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${GH_SA}" \
  --role="roles/iam.serviceAccountUser"

# WIF pool + provider, scoped to your repo
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='YOUR_GH_USER/osllm_backend'"

# Let the GitHub repo impersonate the SA
gcloud iam service-accounts add-iam-policy-binding "$GH_SA" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GH_USER/osllm_backend"
```

Replace `YOUR_GH_USER/osllm_backend` with your actual GitHub `owner/repo`.

### Required GitHub repo secrets

In your repo → Settings → Secrets and variables → Actions, add:

| Name | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project id |
| `GCP_WIF_PROVIDER` | Full resource name, e.g. `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `GCP_SERVICE_ACCOUNT` | `gh-actions-deployer@PROJECT_ID.iam.gserviceaccount.com` |
| `HF_TOKEN` | Your Hugging Face token (used at build time to bake the model) |

### Workflow YAML

Copy this into `.github/workflows/deploy-gcp.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  REPO: osllm
  SERVICE: osllm-backend

permissions:
  contents: read
  id-token: write   # required for OIDC / Workload Identity Federation

jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WIF_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up gcloud
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

      - name: Build image
        run: |
          docker build \
            --build-arg HF_TOKEN="${{ secrets.HF_TOKEN }}" \
            -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${{ github.sha }}" \
            -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest" \
            .

      - name: Push image
        run: |
          docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${{ github.sha }}"
          docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy "${SERVICE}" \
            --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${{ github.sha }}" \
            --region="${REGION}" \
            --platform=managed \
            --cpu=8 \
            --memory=32Gi \
            --cpu-boost \
            --min-instances=0 \
            --max-instances=2 \
            --concurrency=1 \
            --timeout=600 \
            --port=8080 \
            --allow-unauthenticated \
            --set-secrets=HF_TOKEN=hf-token:latest \
            --quiet

      - name: Smoke test
        run: |
          URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
          curl -fsS "${URL}/health"
```

Push the file to `main` and watch the Actions tab — first run builds the ~15 GB image (slow, 15–25 min), subsequent runs are faster thanks to Docker layer caching.

---

## Cost expectations

Cloud Run pricing (us-central1, on-demand, request-based billing):

| Item | Rate | Notes |
|---|---|---|
| CPU | $0.000024 / vCPU-second | Only while serving a request |
| Memory | $0.0000025 / GiB-second | Only while serving a request |
| Requests | $0.40 per million | First 2 M requests/month are free |
| Artifact Registry | $0.10 / GB / month | Image ≈ 15 GB → ~$1.50/month |
| Secret Manager | Free under 6 versions | `hf-token` fits the free tier |

**Realistic monthly cost for hobby use:**
- Idle: ~$1.50 (image storage only)
- 1000 requests × 30s each (8 vCPU / 32 GiB): ~$2 in compute
- **Total: under $5/month**

The single biggest cost lever is request duration. CPU inference for Gemma 4 E2B is ~tens of seconds per response; moving to GPU (when quota lands) cuts this dramatically.

---

## Known limitations

These are issues in the code itself, not in the deployment. Plan to address them before opening the service to real users:

- **In-memory session store** (`main.py:77`) — conversations are lost on cold start and don't shard across instances. Migrate to Redis (Memorystore) or make endpoints stateless.
- **No authentication** — `--allow-unauthenticated` is fine for testing but unsafe for public exposure. Drop the flag (Cloud Run IAM) or add an API-key middleware.
- **Single worker** — `--workers 1` in the `Dockerfile` and `--concurrency=1` in the deploy match: one request per instance at a time. Scale horizontally via `--max-instances`, or move to a real inference server (vLLM, TGI) when you outgrow this.
- **`_processor.parse_response()` return value is discarded** (`ai.py:152`) — likely meant to strip thinking-mode tokens; assign the result or remove the call.
- **`rate_limit_rpm` and `max_content_length`** in `config.yml` are configured but **not enforced** in code.
- **CPU inference is slow** — ~tens of seconds per response. Acceptable for testing, rough for interactive UX.

---

## Upgrading to GPU later

When you're ready and have GPU quota:

1. Request `nvidia_l4_gpu_allocation` quota in [GCP Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas). Filter for "Cloud Run". Default is 0 — that's why your earlier attempt failed.
2. Quota is region-locked — Cloud Run GPU is currently in `us-central1`, `europe-west1`, `europe-west4`, and `asia-southeast1`. Pick one of those for `REGION`.
3. Redeploy with GPU flags:

   ```bash
   gcloud run deploy "$SERVICE" \
     --image="$IMAGE" \
     --region="$REGION" \
     --gpu=1 \
     --gpu-type=nvidia-l4 \
     --cpu=8 \
     --memory=32Gi \
     --no-cpu-throttling \
     --max-instances=1 \
     --min-instances=0 \
     --concurrency=1 \
     --timeout=600 \
     --port=8080 \
     --set-secrets=HF_TOKEN=hf-token:latest
   ```

   Note: GPU instances on Cloud Run require `--no-cpu-throttling`, which means **CPU is always allocated** while the instance is up. Scale-to-zero still works, but per-active-hour cost is higher (~$0.71/hour for L4 + 8 vCPU / 32 GiB at the time of writing).

4. If Cloud Run GPU quota isn't granted, alternatives that don't share the same quota gate: GKE Autopilot with a GPU node pool, or a Vertex AI custom-container endpoint.
