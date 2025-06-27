# Docker and Dependencies Fix

## Issues Fixed

### 1. NVIDIA GPU Driver Error
**Problem**: Docker was trying to use NVIDIA GPU capabilities even on systems without GPU support.

**Solution**: 
- Removed GPU requirements from the main `docker-compose.yml`
- Created a separate `docker-compose.gpu.yml` overlay for GPU systems
- Created two startup scripts:
  - `start_docker_nogpu.sh` - For systems without GPU
  - `start_docker_gpu.sh` - For systems with NVIDIA GPU

### 2. HuggingFace Hub ImportError
**Problem**: `sentence-transformers==2.2.2` was incompatible with newer `huggingface_hub` versions.

**Solution**: 
- Updated `sentence-transformers` to version `2.6.1`
- Pinned `huggingface-hub` to version `0.20.1` for compatibility

## How to Run

### Without GPU (Most Common)
```bash
./start_docker_nogpu.sh
```

### With NVIDIA GPU
```bash
# First ensure you have nvidia-docker2 installed
./start_docker_gpu.sh
```

### Manual Docker Commands

Without GPU:
```bash
docker-compose up --build -d
```

With GPU:
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

## Rebuild Requirements

If you're still experiencing issues, rebuild the containers:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Check Logs

To monitor the services:
```bash
docker-compose logs -f
```

To check specific service:
```bash
docker-compose logs -f emoia-api
```