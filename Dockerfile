FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      git-lfs \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      unzip \
      libgoogle-perftools-dev \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set working directory
WORKDIR /workspace

# Install Torch with timeout and retry mechanism
RUN pip3 install --no-cache-dir --timeout 300 torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || \
    pip3 install --no-cache-dir --timeout 300 torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Inswapper Serverless Worker with error handling
RUN git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git && \
    cd /workspace/runpod-worker-inswapper && \
    pip3 install --no-cache-dir --timeout 300 -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install --no-cache-dir --timeout 300 onnxruntime-gpu

# Download insightface checkpoints with retry mechanism
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p checkpoints/models && \
    cd checkpoints && \
    wget --timeout=300 --tries=3 -O inswapper_128.onnx "https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx?download=true" && \
    cd models && \
    wget --timeout=300 --tries=3 https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir buffalo_l && \
    cd buffalo_l && \
    unzip ../buffalo_l.zip

# Install CodeFormer with error handling
RUN cd /workspace/runpod-worker-inswapper && \
    git lfs install && \
    git clone --depth 1 https://huggingface.co/spaces/sczhou/CodeFormer

# Download CodeFormer weights with retry mechanism
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p CodeFormer/CodeFormer/weights/CodeFormer && \
    wget --timeout=300 --tries=3 -O CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/facelib && \
    wget --timeout=300 --tries=3 -O CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget --timeout=300 --tries=3 -O CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/realesrgan && \
    wget --timeout=300 --tries=3 -O CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"

# Create necessary directories and set permissions
RUN mkdir -p /tmp/inswapper && \
    chmod 755 /tmp/inswapper

# Copy handler to ensure its the latest
COPY --chmod=755 handler.py /workspace/runpod-worker-inswapper/handler.py

# Copy schema files
COPY --chmod=644 schemas/input.py /workspace/runpod-worker-inswapper/schemas/input.py

# Docker container start script
COPY --chmod=755 start.sh /start.sh

# Start the container
ENTRYPOINT /start.sh
