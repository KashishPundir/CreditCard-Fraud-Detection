terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "3.9.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Docker network for the fraud detection application
resource "docker_network" "fraud_network" {
  name = "fraud-detection-network"
}

# Pull the Fraud Detection API image
resource "docker_image" "fraud_api" {
  name         = "kashish1303/fraud-detection-api:latest"
  keep_locally = true
}

# Pull the Streamlit UI image
resource "docker_image" "fraud_ui" {
  name         = "kashish1303/fraud-detection-ui:latest"
  keep_locally = true
}

# Create the API container
resource "docker_container" "fraud_api" {
  name  = "fraud-api"
  image = docker_image.fraud_api.image_id

  ports {
    internal = 8000
    external = 8000
  }

  networks_advanced {
    name = docker_network.fraud_network.name
  }

  restart = "unless-stopped"
}

# Create the UI container
resource "docker_container" "fraud_ui" {
  name  = "fraud-ui"
  image = docker_image.fraud_ui.image_id

  ports {
    internal = 8501
    external = 8501
  }

  env = [
    "API_BASE_URL=http://fraud-api:8000"
  ]

  networks_advanced {
    name = docker_network.fraud_network.name
  }

  restart = "unless-stopped"
}
