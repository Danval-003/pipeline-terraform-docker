# Pipeline ML con Terraform + Docker

Este repositorio define y levanta, con **Terraform**, un pipeline de 3 servicios en **Docker** para procesar datos, entrenar un modelo y evaluar/predicir vía API.

## Servicios

* **preproc (8080):** genera `train/val/test` y un diagrama del pipeline de datos.
* **train (8081):** entrena el modelo, guarda el artefacto (`model.pkl`) y produce reporte de validación.
* **infer (8082):** carga el modelo entrenado, evalúa en *test* y expone `/predict`.

Los servicios comparten red (`ml_net`) y montajes en `/mnt/data`, `/mnt/models`, `/mnt/reports`.

---

## Requisitos

* **Docker Desktop** (Windows/Mac) o Docker Engine (Linux).
* **Terraform** `>= 1.5`.
* **Python 3.10+** para ejecutar `orchestrate.py`.

> En Windows, Terraform se conecta a Docker Desktop por `npipe:////./pipe/docker_engine`.

---

## Estructura (resumen)

```
.
├─ app/
│  ├─ preproc/  # FastAPI: /run, /health
│  ├─ train/    # FastAPI: /train, /health
│  └─ infer/    # FastAPI: /reload, /evaluate_test, /predict, /health
├─ data/        # raw.csv y splits generados (si se usa bind)
├─ src/oxigen_pipeline/   # librería (pipeline, modelos)
├─ orchestrate.py         # orquestación e2e
├─ main.tf                # Terraform (red, imágenes y contenedores)
└─ preproc.json           # config mínima de preproc
```

---

## Configuración rápida

1. (Opcional, recomendado) autenticar Docker Hub en el provider de Terraform para evitar timeouts de pulls:

* Crear `terraform.tfvars`:

```hcl
dockerhub_username = "TU_USUARIO"
dockerhub_token    = "TU_PAT_O_TOKEN"
```

> Si no se desea, se puede quitar el bloque `registry_auth` del `provider "docker"`.

2. Asegurar **data/raw.csv**:

```
data/
└─ raw.csv
```

3. Inicializar y validar:

```bash
terraform init
terraform fmt
terraform validate
```

4. Aplicar:

```bash
terraform apply -auto-approve
```

Esto crea red, imágenes y contenedores de `preproc`, `train` e `infer`.

---

## Verificación rápida

Salud de servicios (cualquiera de las dos variantes):

```bash
# curl
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8082/health
```

```powershell
# PowerShell
Invoke-WebRequest http://localhost:8080/health
Invoke-WebRequest http://localhost:8081/health
Invoke-WebRequest http://localhost:8082/health
```

---

## Ejecutar el pipeline end-to-end

```bash
python orchestrate.py preproc.json XGBoost AQI
```

Salida esperada (resumen real):

```
[preproc] reporte=/mnt/reports/pipeline_diagram.html
[train] val_metrics={'val_r2': 0.8928...}
[train] val_report=/mnt/reports/val_report.html diagram=/mnt/reports/pipeline_model_diagram.html
[infer] reload ok
[infer/evaluate_test] test_metrics={'mse': 684.90, 'mae': 17.98, 'r2': 0.9472}
test_report=/mnt/reports/test_report.html
```

**Artefactos generados:**

* **Datos:** `/mnt/data/train.parquet`, `/mnt/data/val.parquet`, `/mnt/data/test.parquet`
* **Modelo:** `/mnt/models/model.pkl`
* **Reportes HTML:**

  * Preproc: `/mnt/reports/pipeline_diagram.html`
  * Train: `/mnt/reports/val_report.html`, `/mnt/reports/pipeline_model_diagram.html`
  * Infer: `/mnt/reports/test_report.html`

---

## Cómo ver los reportes HTML

### Opción A — copiar desde el contenedor

```bash
docker cp preproc:/mnt/reports/pipeline_diagram.html .
docker cp train:/mnt/reports/val_report.html .
docker cp train:/mnt/reports/pipeline_model_diagram.html .
docker cp infer:/mnt/reports/test_report.html .
```

### Opción B — montar una carpeta local

Si se prefiere que queden en el host sin copiar, cambiar en `main.tf` los montajes de `/mnt/reports` a **bind**:

```hcl
mounts {
  type   = "bind"
  source = abspath("${path.module}/reports") # carpeta local
  target = "/mnt/reports"
}
```

> Crear la carpeta `reports/` antes de aplicar. Luego `terraform apply` para recrear contenedores con el nuevo montaje.

---

## Compartir imágenes Docker sin publicar en un registry

Exportar/importar como archivo:

```bash
# Exportar
docker image save -o ml-train-local.tar ml/train:local

# Importar en otra máquina
docker image load -i ml-train-local.tar
```

(Comprimido)

```bash
docker image save ml/train:local | gzip > ml-train-local.tar.gz
gunzip -c ml-train-local.tar.gz | docker image load
```

---

## Comandos útiles

* Aplicar por partes (cuando la red es inestable):

```bash
terraform apply -target=docker_image.preproc_img -auto-approve
terraform apply -target=docker_image.train_img   -auto-approve
terraform apply -target=docker_image.infer_img   -auto-approve
terraform apply -auto-approve
```

* Forzar reemplazo de un recurso concreto:

```bash
terraform apply -replace=docker_image.train_img -auto-approve
```

* Ver logs:

```bash
docker logs preproc --tail 100
docker logs train   --tail 100
docker logs infer   --tail 100
```

* Destruir todo:

```bash
terraform destroy -auto-approve
```

---

## Notas y recomendaciones

* **Windows paths:** el repo ya usa `abspath("${path.module}/data")` para montar `./data` como bind.
* **Dependencias nativas (LightGBM/XGBoost):** las imágenes instalan `libgomp1` para evitar errores tipo `libgomp.so.1`.
* **Parquet:** se usan `pandas` + `pyarrow` para lectura/escritura.

---

## Configuración mínima de `preproc.json` (ejemplo)

```json
{
  "input_path": "/mnt/data/raw.csv",
  "target_col": "AQI",
  "test_size": 0.2,
  "val_size": 0.2,
  "random_state": 42
}
```

Los valores `test_size`, `val_size` y `random_state` se completan por defecto si no están.

---

## Endpoints (referencia)

* **preproc:**
  `GET /health` — ok
  `POST /run` — genera splits y diagrama (usa `preproc.json`)

* **train:**
  `GET /health` — ok
  `GET /report` — abre `val_report.html`
  `POST /train` — entrena y guarda `model.pkl`

* **infer:**
  `GET /health` — ok / `model-not-loaded`
  `POST /reload` — carga modelo
  `POST /evaluate_test` — métricas + `test_report.html`
  `POST /predict` — predicción para `{"records": [...]}`

---

Con esto, el repo queda listo para levantar la infraestructura con Terraform, ejecutar el pipeline end-to-end y compartir artefactos y/o imágenes Docker de forma portable.
