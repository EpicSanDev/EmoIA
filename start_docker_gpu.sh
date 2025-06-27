#!/bin/bash

echo "🚀 Starting EmoIA with GPU support..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA Docker runtime not found or GPU not accessible."
    echo "Please install nvidia-docker2 and ensure GPU drivers are installed."
    echo "Alternatively, use ./start_docker_nogpu.sh to run without GPU."
    exit 1
fi

echo "✅ GPU detected and accessible!"

# Clean up old containers if they exist
echo "🧹 Cleaning up old containers..."
docker-compose down 2>/dev/null

# Build and start services with GPU support
echo "🔨 Building and starting services with GPU..."
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service status..."
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=50

echo "✅ EmoIA is running with GPU support!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"