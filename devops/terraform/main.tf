terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

resource "docker_image" "experiment" {
  name         = "nginx:alpine"
  keep_locally = true
}

resource "docker_container" "experiment" {
  name  = "terraform-devops-demo"
  image = docker_image.experiment.image_id

  ports {
    internal = 80
    external = 8085
  }
}
