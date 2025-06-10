# Deployment Documentation

## Environment Modes

The application supports two deployment modes:
- Production (default)
- Development (isolated)

## Development Mode Setup

### Local Development

1. Create a `.env.develop` file with development-specific variables:
```bash
cp .env.example .env.develop
# Edit .env.develop with development values
```

2. Start development services:
```bash
docker-compose -f docker-compose.develop.yml up --build
```

Services will be available at:
- Gradio UI: http://localhost:7865
- Dashboard: http://localhost:7862
- Milvus: localhost:19531

### Cloud Development

1. Set up development secrets in GitHub:
   - `GCP_PROJECT_DEV`
   - `GCP_SA_KEY_DEV`
   - `OPENAI_API_KEY_DEV`
   - `OPENAI_IMAGE_MODEL_DEV`
   - `GCS_BUCKET_NAME_DEV`
   - `MILVUS_HOST_DEV`
   - `MILVUS_PORT_DEV`

2. Deploy to development environment:
```bash
./deploy_develop.sh
```

## Changelog

### [2024-03-20] - Development Mode Setup
- Created: `docker-compose.develop.yml` → Isolated development services
- Created: `.env.develop` → Development environment variables
- Created: `deploy_develop.sh` → Development deployment script
- Created: `.github/workflows/deploy_develop.yml` → Development CI/CD workflow
- Created: `docs/deployment.md` → Deployment documentation

### Service Isolation
- Gradio UI: Port 7865 (dev) vs 7860 (prod)
- Dashboard: Port 7862 (dev) vs 7861 (prod)
- Milvus: Port 19531 (dev) vs 19530 (prod)
- Volume directories: `volumes-dev/` vs `volumes/`

### Environment Variables
- All development variables suffixed with `_DEV`
- Separate GCP project and service accounts
- Isolated GCS buckets and API keys

## Production Mode

The original production configuration remains unchanged:
- `docker-compose.yml`
- `deploy_backend.sh`
- `.github/workflows/deploy.yml`

## Best Practices

1. **Local Development**
   - Use `docker-compose.develop.yml` for local testing
   - Never modify production files directly
   - Test changes in development mode first

2. **Cloud Deployment**
   - Development branch deploys to dev environment
   - Main branch deploys to production
   - Use separate service accounts and projects

3. **Data Isolation**
   - Development uses separate volumes
   - Development uses separate GCS buckets
   - Development uses separate Milvus collections

4. **Resource Management**
   - Development uses fewer resources
   - Development has lower instance limits
   - Development uses separate API keys

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