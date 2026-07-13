# Defect Classifier API

FastAPI + ResNet18 (transfer learning) defect classifier for MVTec AD (good
vs defective), deployed on GCP Cloud Run. Same deploy pattern as the
[Fraud Detection API](../Cloud) project (FastAPI + Docker + Terraform +
GitHub Actions CI/CD with Workload Identity Federation) — different task
(image classification + Grad-CAM instead of tabular anomaly detection).

## Setup

```
conda activate torch_env
pip install -r requirements-dev.txt
```

## 1. Get the data

Category archives are public direct downloads (no login) from
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

1. Drop `<category>.tar.xz` files in `data/downloads/`
2. `python data/prepare_mvtec.py bottle hazelnut`

## 2. Train

```
python train.py --category bottle
python train.py --category hazelnut
```

Saves `models/<category>_resnet18.pt` + `models/<category>_metrics.json`
(precision/recall/F1 on a held-out stratified test split).

## 3. Run the API

```
uvicorn app.main:app --reload
```

- `GET /health` — lists loaded categories
- `POST /predict?category=bottle` (multipart image) — `{"defective": bool, "confidence": float}`
- `POST /predict/heatmap?category=bottle` (multipart image) — PNG with Grad-CAM overlay

## 4. Test

```
pytest tests/
```

## Live

`https://defect-classifier-api-874629550296.europe-west1.run.app` (project
`defect-classifier-985319`). Deployed manually first (2026-07-09), then
Terraform + CI/CD wired up and verified end-to-end via a real merge to
`main` (2026-07-09) — `/health`, `/predict` on both categories,
`/predict/heatmap`, 404 on unknown category all checked against the live
URL after the automated deploy, not just CI's green checkmark.

Repo: https://github.com/lele25896/defect-classifier-api

## 5. Deploy (manual, once — done, kept here for reference)

```
gcloud artifacts repositories create defect-classifier-api --repository-format=docker --location=europe-west1
docker build -t europe-west1-docker.pkg.dev/PROJECT/defect-classifier-api/defect-classifier-api:latest .
docker push europe-west1-docker.pkg.dev/PROJECT/defect-classifier-api/defect-classifier-api:latest
gcloud run deploy defect-classifier-api --image europe-west1-docker.pkg.dev/PROJECT/defect-classifier-api/defect-classifier-api:latest --region europe-west1 --allow-unauthenticated --memory 1Gi
```

## 6. CI/CD + Terraform (live)

- `terraform/` — Artifact Registry + Cloud Run v2 + public IAM invoker +
  Cloud Monitoring uptime check on `/health`, GCS remote state. Existing
  manually-created resources were `terraform import`-ed rather than
  recreated (see deep-dive doc for the exact commands).
- `.github/workflows/deploy.yml` — test → `terraform plan`/`apply` → build →
  push → deploy. Repo secrets: `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`,
  `GCP_PROJECT_ID` (Workload Identity Federation, no long-lived key).
- **Model checkpoints are committed to the repo** (`models/*.pt`, ~45MB
  each, under GitHub's 100MB limit) — they're gitignored in most ML repos
  by convention, but here the Docker build needs them. Forgetting this
  once meant the first CI deploy shipped a container with zero models
  loaded (`/predict` 404 on everything) despite a green pipeline.

## 7. Dashboard

Live: https://defect-classifier-dashboard-772rj5jptq-ew.a.run.app

```
pip install -r dashboard/requirements.txt
streamlit run dashboard/dashboard.py
```

Upload an image, pick a category, see prediction + Grad-CAM heatmap side by
side. `dashboard/requirements.txt` is isolated from `requirements.txt`
(no torch) — Streamlit Cloud failed to build on the Fraud Detection project
when the dashboard requirements pulled torch in through a shared file.

Deployed on Cloud Run as a second service (`dashboard/Dockerfile`), same
Terraform/CI pattern as the API — its own Artifact Registry repo, public
Cloud Run service, and `deploy-dashboard` CI job.

## CV line

> Deployed a PyTorch (ResNet18 transfer learning) defect-classification
> model as a containerized FastAPI REST service on **GCP Cloud Run**, with
> **GitHub Actions CI/CD** (build → push → deploy) and **Terraform** IaC;
> includes Grad-CAM interpretability. Free-tier cost (~€0).
