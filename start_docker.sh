#!/bin/bash

# EmoIA Docker Startup Script with Auto GPU Detection

echo "🚀 Starting EmoIA..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data logs models cache models/ollama

# Auto-detect GPU support
echo "🔍 Detecting GPU support..."
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU detected! Starting with GPU support..."
    exec ./start_docker_gpu.sh "$@"
else
    echo "ℹ️  No GPU detected or NVIDIA Docker runtime not installed."
    echo "Starting without GPU support..."
    exec ./start_docker_nogpu.sh "$@"
fi