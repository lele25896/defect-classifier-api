output "service_url" {
  value = google_cloud_run_v2_service.defect_api.uri
}

output "dashboard_url" {
  value = google_cloud_run_v2_service.dashboard.uri
}
