output "api_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Cloud Run URL for the API service."
}

output "raw_bucket_name" {
  value       = google_storage_bucket.raw_zone.name
  description = "Raw-zone storage bucket."
}

output "artifacts_bucket_name" {
  value       = google_storage_bucket.artifacts.name
  description = "Artifacts storage bucket."
}