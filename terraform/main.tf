terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  # backend GCS dal giorno 1 — bucket va creato una tantum a mano prima di
  # `terraform init` (chicken-and-egg: non puoi usare Terraform per creare
  # il backend di Terraform stesso). Sostituisci il nome bucket.
  backend "gcs" {
    bucket = "defect-classifier-985319-tfstate"
    prefix = "cloud-run"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "defect_api" {
  repository_id = var.service_name
  format        = "DOCKER"
  location      = var.region
  depends_on    = [google_project_service.artifact_registry]
}

resource "google_cloud_run_v2_service" "defect_api" {
  name     = var.service_name
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:latest"
      resources {
        limits = { memory = "1Gi" }
      }
      liveness_probe {
        http_get { path = "/health" }
      }
    }
  }

  # Terraform possiede l'infra statica; GitHub Actions possiede le revision
  # (nuove immagini a ogni deploy) — senza questo, `apply` farebbe rollback
  # a `:latest` sovrascrivendo l'ultimo deploy della CI.
  # ponytail: anche con ignore_changes=[template], `terraform plan` mostra
  # un diff cosmetico perpetuo su scaling.manual_instance_count/
  # min_instance_count (0 -> null) — GCP li popola lato server, Terraform
  # non riesce a rappresentarli come "assenti". Non applica mai nulla di
  # distruttivo (nessuna recreate, l'immagine gestita da CI resta intatta).
  # Restringere ignore_changes a path più specifici peggiora la cosa
  # (espone altri default lato server: cpu, probe). Accettato com'è.
  lifecycle {
    ignore_changes = [template, client, client_version]
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.defect_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
