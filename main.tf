terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# variables.tf
variable "dockerhub_username" {
  type        = string
  description = "Usuario de Docker Hub"
}

variable "dockerhub_token" {
  type        = string
  sensitive   = true
  description = "Token/PAT de Docker Hub"
}

# main.tf (provider)
provider "docker" {
  host = "npipe:////./pipe/docker_engine" # Docker Desktop en Windows

  # Evita auth anónima y timeouts contra Docker Hub
  registry_auth {
    address  = "https://index.docker.io/v1/"
    username = var.dockerhub_username
    password = var.dockerhub_token
  }
}

# ---------------------------
# Red y volúmenes compartidos
# ---------------------------
resource "docker_network" "ml_net" { name = "ml_net" }

resource "docker_volume" "ml_data" { name = "ml_data" }
resource "docker_volume" "ml_models" { name = "ml_models" }
resource "docker_volume" "ml_reports" { name = "ml_reports" }

# -----------------------------------------------------
# Imágenes (cada una con SU PROPIO build context)
# -----------------------------------------------------

resource "docker_image" "preproc_img" {
  name = "ml/preproc:local"
  build {
    context     = path.module              # 👈 raíz del repo
    dockerfile  = "app/preproc/Dockerfile" # 👈 Dockerfile del servicio
    pull_parent = false
  }
}

resource "docker_image" "train_img" {
  name = "ml/train:local"
  build {
    context     = path.module
    dockerfile  = "app/train/Dockerfile"
    pull_parent = false
  }
}

resource "docker_image" "infer_img" {
  name = "ml/infer:local"
  build {
    context     = path.module
    dockerfile  = "app/infer/Dockerfile"
    pull_parent = false
  }
}

# ------------------------------------------------
# Contenedor: PREPROC (puerto interno 8080)
# ------------------------------------------------
# PREPROC
resource "docker_container" "preproc" {
  name  = "preproc"
  image = docker_image.preproc_img.image_id

  networks_advanced {
    name = docker_network.ml_net.name
  }

  mounts {
    type   = "bind"
    source = abspath("${path.module}/data")  # carpeta local ./data
    target = "/mnt/data"
  }

  mounts {
    type   = "volume"
    source = docker_volume.ml_models.name
    target = "/mnt/models"
  }
  mounts {
    type   = "bind"
    source = abspath("${path.module}/reports")  # -> C:/Users/.../model_pipeline/reports
    target = "/mnt/reports"
  }

  ports {
    internal = 8080
    external = 8080
  }

  healthcheck {
    test     = ["CMD", "python", "-c", "import socket; s=socket.create_connection(('127.0.0.1',8080),2); s.close()"]
    interval = "10s"
    timeout  = "3s"
    retries  = 3
  }

  depends_on = [docker_network.ml_net, docker_image.preproc_img]
}

# TRAIN
resource "docker_container" "train" {
  name  = "train"
  image = docker_image.train_img.image_id

  networks_advanced {
    name = docker_network.ml_net.name
  }

  mounts {
    type   = "bind"
    source = abspath("${path.module}/data")
    target = "/mnt/data"
  }
  mounts {
    type   = "volume"
    source = docker_volume.ml_models.name
    target = "/mnt/models"
  }
  mounts {
    type   = "bind"
    source = abspath("${path.module}/reports")  # -> C:/Users/.../model_pipeline/reports
    target = "/mnt/reports"
  }

  ports {
    internal = 8081
    external = 8081
  }

  healthcheck {
    test     = ["CMD", "python", "-c", "import socket; s=socket.create_connection(('127.0.0.1',8081),2); s.close()"]
    interval = "10s"
    timeout  = "3s"
    retries  = 3
  }

  depends_on = [docker_network.ml_net, docker_image.train_img]
}

# INFER
resource "docker_container" "infer" {
  name  = "infer"
  image = docker_image.infer_img.image_id

  networks_advanced { name = docker_network.ml_net.name }

  # YA tenés el de modelos:
  mounts {
    type   = "volume"
    source = docker_volume.ml_models.name
    target = "/mnt/models"
  }

  # 👇 Necesario para leer /mnt/data/test.parquet
  mounts {
    type   = "bind"
    source = abspath("${path.module}/data")
    target = "/mnt/data"
  }

  # 👇 Opcional pero recomendable para persistir el reporte /mnt/reports/test_report.html
  mounts {
    type   = "bind"
    source = abspath("${path.module}/reports")  # -> C:/Users/.../model_pipeline/reports
    target = "/mnt/reports"
  }

  ports {
    internal = 8082
    external = 8082
  }

  healthcheck {
    test     = ["CMD", "python", "-c", "import socket; s=socket.create_connection(('127.0.0.1',8082),2); s.close()"]
    interval = "10s"
    timeout  = "3s"
    retries  = 3
  }

  depends_on = [docker_network.ml_net, docker_image.infer_img]
}


