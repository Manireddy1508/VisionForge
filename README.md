# Cloud Image Generation Backend (Google Imagen API)

## Overview
This backend provides a Gradio UI for prompt-enhanced image generation using Google Cloud's Imagen API (Vertex AI). All generated images are stored in Google Cloud Storage (GCS) and returned with signed URLs for secure access.

---

## Prerequisites
- Google Cloud project with billing enabled
- Vertex AI and Cloud Run APIs enabled
- Google Container Registry enabled
- Service account with:
  - `roles/aiplatform.user`
  - `roles/storage.objectAdmin`
- Docker installed
- `gcloud` CLI installed and authenticated

---

## Environment Variables
Set these in your `.env` file or Cloud Run environment:

- `OPENAI_API_KEY` (for prompt enhancement)
- `GOOGLE_IMAGE_MODEL` (default: `imagen-3.0-generate-002`)
- `GCS_BUCKET_NAME` (your GCS bucket for outputs)
- `GOOGLE_CLOUD_PROJECT` (your GCP project ID)
- `GOOGLE_CLOUD_LOCATION` (default: `us-central1`)

---

## Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python src/app.py
   ```
3. Access Gradio UI at [http://localhost:7860](http://localhost:7860)

---

## Docker Build & Run
1. Build the Docker image:
   ```bash
   docker build -t cloud-image-gen .
   ```
2. Run locally:
   ```bash
   docker run --env-file .env -p 7860:7860 cloud-image-gen
   ```

---

## Deploy to Google Cloud Run
1. Edit `deploy_backend.sh` with your project and bucket info, or export as env vars:
   ```bash
   export PROJECT_ID=your-gcp-project
   export REGION=us-central1
   export SERVICE_NAME=cloud-image-gen
   export GCS_BUCKET=your-bucket-name
   ./deploy_backend.sh
   ```
2. The script will:
   - Build and push the Docker image
   - Deploy to Cloud Run with 4Gi memory, 2 CPUs, 600s timeout
   - Allow unauthenticated access
   - Print the service URL

---

## Testing the Service
- Open the Cloud Run service URL in your browser.
- Enter a prompt and (optionally) reference images.
- Generate enhanced prompts, then generate images.
- Each generated image is stored in GCS and a signed URL is used for display/download.

---

## Testing Milvus Integration with Docker

To test the Milvus integration, follow these steps:

1. **Ensure Docker is installed**: Make sure you have Docker installed on your system.

2. **Build the Docker image**: Run the following command in the root directory of your project to build the Docker image:
   ```bash
   docker build -t milvus-integration .
   ```

3. **Run the Docker container**: Use the following command to run the container:
   ```bash
   docker run -p 7860:7860 milvus-integration
   ```

4. **Access the application**: Open your web browser and go to `http://localhost:7860` to access the application.

5. **Check logs**: You can check the logs in the `logs` directory inside the container. You can also view the logs directly in the terminal where you ran the Docker container.

6. **Clean up**: After testing, you can stop the container by pressing `Ctrl + C` in the terminal where the container is running.

This setup will allow you to test the Milvus integration in an isolated environment.

---

## Notes
- All secrets/configs are managed via environment variables.
- Prompt enhancement and image generation are modular and separated.
- Signed URLs for GCS outputs are valid for 1 hour by default.
- For production, restrict service account permissions as needed. 