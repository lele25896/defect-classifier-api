variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "service_name" {
  type    = string
  default = "defect-classifier-api"
}

variable "dashboard_service_name" {
  type    = string
  default = "defect-classifier-dashboard"
}
