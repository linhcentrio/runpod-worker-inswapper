#!/usr/bin/env bash

set -e  # Exit on any error

echo "🚀 Worker Initiated"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ CUDA not detected, will use CPU"
fi

echo "📁 Symlinking files from Network Volume"
ln -sf /runpod-volume /workspace
rm -rf /root/.cache
rm -rf /root/.ifnude
rm -rf /root/.insightface
ln -sf /runpod-volume/.cache /root/.cache
ln -sf /runpod-volume/.ifnude /root/.ifnude
ln -sf /runpod-volume/.insightface /root/.insightface

# Verify required files exist
echo "🔍 Verifying required files..."
if [ ! -f "/workspace/runpod-worker-inswapper/handler.py" ]; then
    echo "❌ handler.py not found"
    exit 1
fi

if [ ! -f "/workspace/runpod-worker-inswapper/checkpoints/inswapper_128.onnx" ]; then
    echo "❌ inswapper_128.onnx not found"
    exit 1
fi

if [ ! -d "/workspace/runpod-worker-inswapper/CodeFormer" ]; then
    echo "❌ CodeFormer directory not found"
    exit 1
fi

echo "✅ All required files verified"

echo "🐍 Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /workspace/runpod-worker-inswapper

# Start the handler with error handling
python3 -u handler.py
