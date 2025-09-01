FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/work/.cache/huggingface \
    TRANSFORMERS_CACHE=/work/.cache/huggingface \
    HF_DATASETS_CACHE=/work/.cache/huggingface

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
