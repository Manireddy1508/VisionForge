#!/bin/bash
set -e

# === Config ===
PROJECT_ID=${PROJECT_ID:-your-gcp-project-dev}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-cloud-image-gen-dev}
IMAGE_NAME=gcr.io/$PROJECT_ID/$SERVICE_NAME:develop
GCS_BUCKET=${GCS_BUCKET:-your-dev-bucket-name}

# === Build Docker Image ===
echo "[1/4] Building Docker image..."
docker build -t $IMAGE_NAME ./

echo "[2/4] Pushing image to Google Container Registry..."
gcloud auth configure-docker
docker push $IMAGE_NAME

# === Deploy to Cloud Run ===
echo "[3/4] Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --region $REGION \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=$GCS_BUCKET \
  --set-env-vars GOOGLE_IMAGE_MODEL=imagen-3.0-generate-002 \
  --set-env-vars GRADIO_SERVER_NAME=0.0.0.0 \
  --set-env-vars GRADIO_SERVER_PORT=7860 \
  --set-env-vars ENVIRONMENT=develop

echo "[4/4] Deployment complete!"
echo "Cloud Run service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)' 