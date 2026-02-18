# RIAWELC — Cloud Run Deployment

Deploy the RIAWELC API to Google Cloud Run using `cloudrun.sh`.

## Prerequisites

1. **Google Cloud SDK** (gcloud CLI) installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Docker** installed and running locally.

3. **Artifact Registry** repository created in your GCP project:
   ```bash
   gcloud artifacts repositories create riawelc \
       --repository-format=docker \
       --location=europe-west3 \
       --description="RIAWELC container images"
   ```

4. **IAM permissions** — the deploying account needs:
   - `roles/run.admin` (Cloud Run Admin)
   - `roles/artifactregistry.writer` (push images)
   - `roles/iam.serviceAccountUser` (act as the Cloud Run service account)

5. **Environment file** — a `.env` file in the project root with `RIAWELC_*` variables.
   See `.env.example` for the full list. At minimum:
   ```
   RIAWELC_MODEL_PATH=outputs/models/checkpoints/efficientnetb0/v1/fine_tune/best.keras
   RIAWELC_SEG_BASELINE_PATH=outputs/models/checkpoints/unet_efficientnetb0/v1/best.keras
   RIAWELC_SEG_AUGMENTED_PATH=outputs/models/checkpoints/unet_efficientnetb0_augmented/v1/best.keras
   RIAWELC_API_KEY=<your-secret-api-key>
   ```

## Usage

```bash
# Basic deployment (defaults: region=europe-west3, service=riawelc-api)
./deploy/cloudrun.sh --project my-gcp-project

# Full options
./deploy/cloudrun.sh \
    --project my-gcp-project \
    --region us-central1 \
    --service-name riawelc-staging \
    --image-tag v1.0.0 \
    --env-file .env.production

# Show help
./deploy/cloudrun.sh --help
```

## What the Script Does

1. **Configures Docker** for Artifact Registry authentication.
2. **Builds** the Docker image from the project root `Dockerfile`.
3. **Pushes** the image to `REGION-docker.pkg.dev/PROJECT/riawelc/SERVICE:TAG`.
4. **Deploys** to Cloud Run with:
   - 4 GiB memory, 2 vCPUs
   - 0 minimum / 3 maximum instances (scale to zero)
   - Startup probe on `/health` (period 10s, timeout 5s, 6 failures)
   - Liveness probe on `/health` (period 30s, timeout 5s, 3 failures)
   - All `RIAWELC_*` and `OTEL_*` env vars from `.env`
   - IAM authentication required (`--no-allow-unauthenticated`)
   - Startup CPU boost enabled
5. **Prints** the deployed service URL and a sample `curl` command.

## Resource Configuration

| Setting            | Value        | Rationale                                    |
|--------------------|--------------|----------------------------------------------|
| Memory             | 4 GiB        | TensorFlow model loading + inference buffer  |
| CPU                | 2 vCPUs      | Concurrent preprocessing + inference         |
| Min instances      | 0            | Scale to zero when idle (cost saving)        |
| Max instances      | 3            | Limit concurrent model replicas              |
| Startup probe      | `/health`    | Ensures readiness before receiving traffic   |
| Liveness probe     | `/health`    | Restarts unhealthy containers                |
| Authentication     | IAM-only     | No public access; use `roles/run.invoker`    |

## Granting Access

Since the service uses `--no-allow-unauthenticated`, callers must be
authorized via IAM:

```bash
# Grant a user
gcloud run services add-iam-policy-binding riawelc-api \
    --region=europe-west3 \
    --member="user:someone@example.com" \
    --role="roles/run.invoker"

# Grant a service account (for CI/CD or other services)
gcloud run services add-iam-policy-binding riawelc-api \
    --region=europe-west3 \
    --member="serviceAccount:ci-bot@my-project.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

## Testing the Deployment

```bash
# Get an identity token (for IAM-authenticated access)
TOKEN=$(gcloud auth print-identity-token)

# Health check
curl -H "Authorization: Bearer ${TOKEN}" https://SERVICE_URL/health

# Classification prediction (with API key if configured)
curl -H "Authorization: Bearer ${TOKEN}" \
     -H "X-API-Key: your-api-key" \
     -F "file=@test_image.png" \
     https://SERVICE_URL/predict
```
