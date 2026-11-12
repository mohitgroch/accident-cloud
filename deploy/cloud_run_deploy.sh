#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Run Deployment Script
# Team Member 4 — Cloud & AI Deployment
# ==============================================================================
set -e

PROJECT_ID="bmsce-smart-vehicle"
REGION="asia-south1"
SERVICE_NAME="accident-detection-cloud"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=== Building Container Image ==="
docker build -t ${IMAGE_TAG} -f deploy/Dockerfile .

echo "=== Pushing to Google Container Registry / Artifact Registry ==="
docker push ${IMAGE_TAG}

echo "=== Deploying to Google Cloud Run ==="
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_TAG} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 10s \
    --min-instances 1 \
    --max-instances 5 \
    --set-env-vars INFERENCE_MODE=stub,TIMEOUT_SECONDS=5.0

echo "=== Deployment Successful! ==="
gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)'

