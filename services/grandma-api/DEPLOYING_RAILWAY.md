# Deploying Grandma API to Railway

This guide covers deploying the grandma-api service to Railway.

## Prerequisites

- Railway account ([railway.app](https://railway.app))
- GitHub repository with the monorepo
- OpenAI API key with Whisper access

## Deployment Steps

### 1. Create New Project

1. Log in to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose the repository containing this monorepo

### 2. Configure Build Settings

Set the following build settings:

| Setting | Value |
|---------|-------|
| Root Directory | `services/grandma-api` |
| Build Command | (use Dockerfile) |
| Dockerfile Path | `services/grandma-api/Dockerfile` |

**Important:** The Dockerfile expects to be built from the monorepo root with context `../..` to access shared libs.

### 3. Configure Environment Variables

Add the following environment variables in Railway:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for Whisper |
| `API_TOKEN` | Yes | Authentication token for the API |
| `PORT` | No | Railway sets this automatically |
| `LOG_LEVEL` | No | Default: `INFO` |
| `LOG_FORMAT` | No | Default: `json` |

### 4. Configure Health Checks

Railway will automatically use the `/healthz` endpoint for health checks.

Configure in service settings:
- Health Check Path: `/healthz`
- Health Check Timeout: 10 seconds

### 5. Configure Networking

1. Go to Settings > Networking
2. Enable "Public Networking"
3. Generate a domain or configure custom domain

### 6. Deploy

Railway will automatically build and deploy when you push to the configured branch.

Monitor the deployment in the Railway dashboard:
- Build logs show Docker build progress
- Deploy logs show application startup

## Dockerfile Context

The Dockerfile is designed for monorepo deployment:

```dockerfile
# Built from monorepo root
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/grandma-api/
COPY ${APP_DIR}/src ./services/grandma-api/src
COPY libs ./libs
```

This copies:
- The grandma-api service
- Shared libs (`platform_core`, `platform_stt`)

## Monitoring

### Logs

View logs in Railway dashboard or CLI:

```bash
railway logs
```

Logs are structured JSON format for easy parsing:

```json
{"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "Translating audio", "audio_filename": "recording.webm", "size_bytes": 12345}
```

### Health Checks

Monitor `/healthz` endpoint:

```bash
curl https://your-app.railway.app/healthz
```

## Scaling

Railway automatically scales based on load. For manual configuration:

1. Go to Settings > Resources
2. Adjust memory and CPU limits
3. Configure replica count if needed

## Troubleshooting

### Build Fails

1. Check that the Dockerfile context is correct
2. Verify libs directory structure
3. Check build logs for missing dependencies

### Runtime Errors

1. Verify `OPENAI_API_KEY` is set correctly
2. Check `API_TOKEN` matches what clients use
3. Review application logs for errors

### Connection Refused

1. Ensure `PORT` is not set (Railway provides it)
2. Verify health check is passing
3. Check networking settings

## Cost Optimization

- Use Railway's free tier for development
- Monitor usage in Railway dashboard
- Set resource limits to control costs

## Security Considerations

1. Use Railway's secrets management for API keys
2. Enable HTTPS (automatic on Railway)
3. Use strong, unique `API_TOKEN`
4. Monitor logs for unauthorized access attempts
