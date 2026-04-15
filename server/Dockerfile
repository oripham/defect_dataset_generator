FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# FastAPI deps (lightweight, cached early)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

# AI/ML deps — excludes torch (base image already has torch 2.1.0+cu121).
COPY server/requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

# App code
COPY scripts/ ./scripts/
COPY engines/ ./engines/
COPY server/main.py ./main.py
COPY server/download_models.py ./download_models.py

ENV HF_HOME=/workspace/models
ENV TRANSFORMERS_CACHE=/workspace/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "3600", "--log-level", "info"]
