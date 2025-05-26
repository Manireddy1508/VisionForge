# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISION_DEVICE=cpu \
    VISION_MAX_TOKENS=120 \
    MODEL_CACHE_DIR=/app/model_cache \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_USE_FAST=True

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Create model cache directory with proper permissions
RUN mkdir -p ${MODEL_CACHE_DIR} && \
    chmod 777 ${MODEL_CACHE_DIR}

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and env file first to ensure proper loading
COPY .env .env
COPY . .

# Set environment variables for production
ENV GRADIO_SERVER_NAME=0.0.0.0

# Expose Gradio port
EXPOSE 7860

# Entrypoint
CMD ["python", "src/app.py"] 