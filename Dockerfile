# Echo Studio — CUDA image. Target: an NVIDIA GPU box with nvidia-container-toolkit.
# Build:  docker build -t echo-studio .
# Run:    docker run --gpus all -p 8000:8000 \
#           -v $PWD/models:/app/models -v $PWD/outputs:/app/outputs echo-studio
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# System deps for sentencepiece / pillow etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch already present in the base image; install the rest.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ECHO_HOST=0.0.0.0 ECHO_PORT=8000 ECHO_DEVICE=cuda
EXPOSE 8000

# Download the model at container start if the mounted volume is empty, then serve.
CMD bash -c "[ -f models/chroma-q4.gguf ] || python scripts/download_models.py; \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000"
