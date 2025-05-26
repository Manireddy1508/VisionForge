# Deployment Guide

This document outlines the CI/CD pipeline and deployment process for the AI Image Generation System.

## CI/CD Pipeline Overview

The pipeline consists of two main workflows:

1. **Build Workflow** (`.github/workflows/build.yml`)
   - Triggers on push/PR to main branch
   - Authenticates with Google Cloud using service account JSON
   - Builds Docker images for:
     - Backend service (Gradio + GPT prompt enhancer)
     - Generator service (OpenAI GPT model API wrapper)
   - Pushes images to Google Container Registry (GCR)
   - Tags images with commit SHA and latest (for main branch)

2. **Deploy Workflow** (`.github/workflows/deploy.yml`)
   - Runs after successful build workflow
   - Deploys both services to Cloud Run
   - Configures environment variables and service settings
   - Outputs service URLs in workflow summary

## Required GitHub Secrets

The following secrets must be configured in your GitHub repository:

### Google Cloud Authentication
- `GCP_SA_KEY`: Full contents of service account JSON file
- `GCP_PROJECT`: Your Google Cloud project ID

### OpenAI Configuration
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_IMAGE_MODEL`: OpenAI image model name (e.g., "dall-e-3")

### Milvus Configuration
- `MILVUS_HOST`: Milvus server host address
- `MILVUS_PORT`: Milvus server port (default: 19530)

### Google Cloud Storage
- `GCS_BUCKET_NAME`: GCS bucket name for storing images

## Deployment Process

1. **Initial Setup**
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/your-repo.git
   cd your-repo

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Local Testing**
   ```bash
   # Run tests
   pytest tests/

   # Run linting
   flake8 src/ tests/
   black --check src/ tests/
   mypy src/ tests/
   ```

3. **Deployment Verification**
   - Check GitHub Actions workflow status
   - Verify Cloud Run service URLs
   - Test the deployed services

## Monitoring

### GitHub Actions
1. **View Workflow Status**
   - Go to repository → Actions
   - Select the workflow run
   - Check each step's logs

2. **Build Workflow**
   - Monitor Docker image builds
   - Verify image tags and pushes
   - Check for any build errors

3. **Deploy Workflow**
   - Verify service deployment status
   - Check environment variable configuration
   - Note the deployed service URLs

### Cloud Run
1. **View Service Logs**
   ```bash
   # View backend service logs
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=cloud-gen-backend" --limit 50

   # View generator service logs
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=cloud-gen-generator" --limit 50
   ```

2. **Monitor Service Health**
   - Check service status in Cloud Console
   - Monitor request latency and errors
   - Review resource utilization

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Docker build logs in GitHub Actions
   - Verify Dockerfile syntax and dependencies
   - Ensure all required files are present

2. **Deployment Failures**
   - Verify GCP service account permissions
   - Check Cloud Run logs for errors
   - Ensure all environment variables are set

3. **Service Issues**
   - Check Cloud Run logs for runtime errors
   - Verify Milvus connection
   - Test OpenAI API access

## Security Best Practices

1. **Service Account Management**
   - Use a dedicated service account for CI/CD
   - Grant minimum required permissions:
     - Cloud Run Admin
     - Storage Admin
     - Container Registry Writer
   - Rotate service account keys regularly
   - Never commit service account JSON to repository

2. **Environment Variables**
   - Keep secrets in GitHub Secrets
   - Use `.env.example` for documentation
   - Never commit `.env` files

3. **API Keys**
   - Rotate OpenAI API keys regularly
   - Monitor API usage and quotas
   - Set up alerts for unusual activity

## Maintenance

1. **Regular Updates**
   - Keep dependencies updated
   - Monitor security advisories
   - Update service account keys every 90 days

2. **Monitoring**
   - Set up Cloud Monitoring alerts
   - Monitor API quotas
   - Track error rates and latency

3. **Backup**
   - Regularly backup Milvus data
   - Export Cloud Run configurations
   - Document deployment procedures

## Support

For issues or questions:
1. Check the troubleshooting guide
2. Review Cloud Run logs
3. Contact the development team 