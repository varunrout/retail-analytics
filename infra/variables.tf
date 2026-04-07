variable "project_id" {
  type        = string
  description = "GCP project identifier."
}

variable "region" {
  type        = string
  description = "Primary deployment region."
  default     = "europe-west2"
}

variable "environment" {
  type        = string
  description = "Deployment environment name."
  default     = "dev"
}

variable "container_image" {
  type        = string
  description = "Container image for the API deployment."
}