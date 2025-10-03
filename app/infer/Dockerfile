# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 💡 OpenMP runtime para LightGBM/XGBoost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/oxigen_pipeline ./oxigen_pipeline
COPY app/infer ./app/infer

EXPOSE 8082
CMD ["uvicorn", "app.infer.main:app", "--host", "0.0.0.0", "--port", "8082"]
