terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "raw_zone" {
  name                        = "${var.project_id}-${var.environment}-hb360-raw"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-${var.environment}-hb360-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}

resource "google_service_account" "platform" {
  account_id   = "hb360-platform-${var.environment}"
  display_name = "HealthBeauty360 platform service account"
}

resource "google_project_iam_member" "run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.platform.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name     = "hb360-api-${var.environment}"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.platform.email

    containers {
      image = var.container_image

      env {
        name  = "DEMO_MODE"
        value = "true"
      }

      env {
        name  = "PORT"
        value = "8080"
      }
    }
  }
}

resource "google_cloud_scheduler_job" "daily_pipeline" {
  name        = "hb360-daily-${var.environment}"
  description = "Daily HealthBeauty360 refresh"
  schedule    = "0 6 * * *"
  region      = var.region

  http_target {
    uri         = google_cloud_run_v2_service.api.uri
    http_method = "GET"
  }
}