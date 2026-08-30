
# MLOps Pipeline for Chest CT-Scan Cancer Classification

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline 
7. Update the main.py
8. Update the dvc.yaml 

## How to run?

```bash
conda create -n chest python=3.8 -y
```

```bash
conda activate chest
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

The app is served with FastAPI + Uvicorn on port `8080`. Interactive API docs (Swagger UI) are auto-generated at `/docs`.

## Testing & linting

```bash
pip install -r requirements-dev.txt
flake8 src tests app.py scripts
pytest -q
```

`tests/` covers `cnnClassifier.utils.common`, `ConfigurationManager`, `DataValidation` (config/params parsing, directory creation, JSON round-trips, base64 image encode/decode, corrupt-file/imbalance detection), and the full FastAPI surface in `test_api.py` (health/readiness, prediction success and error paths, the `/train` background task, Prometheus metrics) — all without needing TensorFlow, a downloaded dataset, or a trained model, since the API tests monkeypatch the model-loading step. These same two commands are what the `integration` job in `.github/workflows/main.yaml` runs on every push.

## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## Pipeline stages

`dvc.yaml` runs five stages in order (`dvc dag` shows the graph):

1. **Data ingestion** — downloads and unzips the dataset.
2. **Data validation** — checks every image actually decodes, reports the per-class counts, and flags class imbalance above `MAX_IMBALANCE_RATIO` (`params.yaml`). Fails the pipeline if any file is corrupt or a class is missing. Output: `artifacts/data_validation/validation_report.json`.
3. **Prepare base model** — VGG16 transfer learning setup.
4. **Training** — fine-tunes on the validated dataset.
5. **Evaluation** — see below.

## Evaluation & experiment tracking

The evaluation stage does more than report accuracy:

- **Confusion matrix + per-class precision/recall/F1** via scikit-learn, since for a cancer/normal classifier, accuracy alone hides whether the model is actually catching the cancer class. Saved to `artifacts/evaluation/confusion_matrix.png` and `classification_report.json`, and tracked as a DVC plot/metric (`dvc plots show`).
- **MLflow tracking** (via DagsHub) logs params, the full metric breakdown, and the confusion matrix/report as artifacts for every run.
- **DVC ↔ MLflow reproducibility**: each MLflow run is tagged with the current git commit and has `dvc.lock` attached as an artifact, so any run's numbers can be traced back to the exact code + data + model version that produced them — `git checkout <commit> && dvc repro` reproduces it.

## Data validation & class imbalance

Before training ever sees the data, the `data_validation` stage opens every image to confirm it decodes, counts images per class, and computes an imbalance ratio (majority class size ÷ minority class size). This is the kind of check that's easy to skip and expensive to skip on a medical dataset — a silently corrupt file or a heavily skewed class split can quietly wreck a "high accuracy" result. See `tests/test_data_validation.py` for the behavior under corrupt files, missing directories, and skewed class counts.

## Serving API

`app.py` is a FastAPI app (Flask was replaced by it) exposing:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the frontend (`templates/index.html`). |
| `/health` | GET | Liveness probe — process is up. |
| `/ready` | GET | Readiness probe — 503 until the model has actually loaded. |
| `/predict` | POST | `{"image": "<base64>"}` → `{"label": "...", "confidence": 91.2}`. Returns 400 on malformed/non-image input, 503 if the model isn't loaded. |
| `/train` | POST | Kicks off `dvc repro` as a background task and returns immediately (the old Flask route blocked the whole request until training finished). |
| `/metrics` | GET | Prometheus exposition format — see [Monitoring](#monitoring-prometheus--grafana). |
| `/docs` | GET | Auto-generated Swagger UI. |

Each request gets its own temp file for the uploaded image instead of a single shared filename — the original Flask version wrote every request to the same `inputImage.jpg`, which is a race condition under concurrent requests.

## Monitoring (Prometheus + Grafana)

```bash
docker compose up --build
```

Brings up three containers: the API (`:8080`), Prometheus (`:9090`) scraping its `/metrics` endpoint every 10s, and Grafana (`:3000`, login `admin`/`admin`) with a pre-provisioned dashboard (`monitoring/grafana/dashboards/cnn-classifier-dashboard.json`) showing request rate, error rate, p95 request/inference latency, and — the metric that matters most for a deployed classifier — **predicted-class distribution over time**, since a sudden shift there is an early drift signal you can compute without any ground-truth labels.

> I wrote and syntax-validated the compose file, Prometheus scrape config, and Grafana provisioning/dashboard JSON, and verified `/metrics` emits real Prometheus-format output against the running app (see below) — but couldn't actually run `docker compose up` myself, since there's no Docker daemon in this environment. Worth a real end-to-end check on your machine before you rely on it.

## Inference benchmarking

`python scripts/benchmark_inference.py` compares candidate backbones (VGG16, ResNet50, MobileNetV2) on **inference latency, throughput, and parameter count** — not accuracy, since a fair accuracy comparison would need fine-tuning each one on the real dataset, which needs a GPU this project doesn't have access to. All three use the same untrained 2-class classification head, since weight values don't affect latency, only architecture does.

Measured on this machine (CPU only, no GPU):

| Model | Params | p50 latency, batch=1 | Throughput, batch=16 |
|---|---|---|---|
| VGG16 (current) | 14.8M | 104.5 ms | 23.6 img/s |
| ResNet50 | 23.8M | 68.4 ms | 53.3 img/s |
| MobileNetV2 | 2.4M | 48.1 ms | 161.2 img/s |

Full results (all batch sizes, mean/p50/p95) are in `benchmarks/inference_benchmark.json`. Takeaway: VGG16 is the slowest and largest of the three here — a real reason to eventually benchmark accuracy too, since if MobileNetV2 or ResNet50 fine-tune to comparable accuracy, either would be the better production choice for latency-sensitive serving.


## AWS CI/CD Deployment with GitHub Actions

Every push to `main` runs `.github/workflows/main.yaml`, which has three jobs:
1. **Continuous Integration** — installs `requirements-dev.txt` and runs `flake8 src tests app.py scripts` and `pytest -q` (see [Testing & linting](#testing--linting)).
2. **Continuous Delivery** — builds the Docker image and pushes it to ECR.
3. **Continuous Deployment** — (self-hosted runner only, see steps below) pulls and runs the image on EC2, then polls `/health` until the container reports healthy before cleaning up the old image.

The deployment steps below (2–7) are only needed if you want the CD job to actually land on an EC2 instance; the CI job runs regardless.

### 1. Create an IAM user for deployment

Create an IAM user with these policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

Save the generated Access Key ID and Secret Access Key.

**Deployment flow this sets up:**
1. Build a Docker image of the source code
2. Push the image to ECR
3. Launch an EC2 instance
4. Pull the image from ECR onto EC2
5. Run the Docker image container on EC2

### 2. Create an ECR repository

Create an ECR repo to store/pull the Docker image, e.g.:
```
343218195178.dkr.ecr.us-east-1.amazonaws.com/cnnclassifier
```

### 3. Launch an EC2 instance (Ubuntu)

### 4. Install Docker on the EC2 instance

```bash
sudo apt-get update -y
sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 5. Configure the EC2 instance as a self-hosted GitHub Actions runner

In your GitHub repo: **Settings > Actions > Runners > New self-hosted runner**, choose Linux, and run the commands it gives you on the EC2 instance.

### 6. Set up GitHub repository secrets

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | from the IAM user |
| `AWS_SECRET_ACCESS_KEY` | from the IAM user |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY_NAME` | e.g. `cnnclassifier` (repo name only, not the full URI) |

### 7. Push to `main`

Pushing to `main` triggers `.github/workflows/main.yaml`, which builds the Docker image, pushes it to ECR, then (via the self-hosted runner on your EC2 instance) pulls and runs it on port `8080`.

Make sure the EC2 instance's security group allows inbound traffic on port `8080`. The app will then be reachable at `http://<EC2-public-ip>:8080`.

**Note:** the `/train` route inside the app kicks off `dvc repro` in the background, which needs outbound internet access (to download data and reach DagsHub for MLflow tracking) plus DagsHub auth configured on the instance. This isn't required for `/predict` (the main serving path) to work.
## Free Deployment: Hugging Face Spaces

This repo is also set up to deploy for free on [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK, free CPU tier: 2 vCPU / 16GB RAM, no credit card required).

The `hf-space` branch is a squashed, ready-to-push snapshot of this repo (model weights tracked with Git LFS, and a README with the Spaces metadata header) so it can be pushed directly as a new Space.

### Steps

1. Create a free account at https://huggingface.co and create a new Space:
   - SDK: **Docker**
   - Visibility: your choice (public/private)
2. Add the Space as a git remote and push the `hf-space` branch to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space hf-space:main
   ```
   You'll be prompted for your Hugging Face username and an [access token](https://huggingface.co/settings/tokens) (use it in place of a password).
3. The Space will build the Dockerfile and start serving on port 8080 automatically. Your app will be live at:
   ```
   https://huggingface.co/spaces/<your-username>/<space-name>
   ```

To update the Space later, rebuild the `hf-space` branch from `main` (or just push new changes into it) and `git push space hf-space:main` again.
