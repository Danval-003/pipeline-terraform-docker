# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# deps primero para cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia tu librería y la app
COPY ./src/oxigen_pipeline ./oxigen_pipeline
COPY app/preproc ./app/preproc

EXPOSE 8080

# opcional: healthcheck simple (el de Terraform ya usa socket)
# HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD python -c "import socket; s=socket.create_connection(('127.0.0.1',8080),2); s.close()" || exit 1

CMD ["uvicorn", "app.preproc.main:app", "--host", "0.0.0.0", "--port", "8080"]
