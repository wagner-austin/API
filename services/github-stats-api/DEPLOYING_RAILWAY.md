# Deploying GitHub Stats API to Railway

This guide covers deploying the github-stats-api service to Railway.

## Prerequisites

- Railway account ([railway.app](https://railway.app))
- GitHub repository with the monorepo
- GitHub personal access token with `read:user` scope

## Deployment Steps

### 1. Create New Project

1. Log in to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose the repository containing this monorepo

### 2. Configure Environment Variables

Add the following environment variables in Railway:

| Variable | Required | Description |
|----------|----------|-------------|
| `RAILWAY_DOCKERFILE_PATH` | Yes | `services/github-stats-api/Dockerfile` |
| `GITHUB_TOKEN` | Yes | GitHub PAT with `read:user` scope |
| `CACHE_TTL_SECONDS` | No | Response cache duration (default: 1800) |
| `PORT` | No | Railway sets this automatically |

**Important:** The `RAILWAY_DOCKERFILE_PATH` variable tells Railway to use the service's Dockerfile while building from the monorepo root. This is required because the Dockerfile needs access to `libs/platform_core`.

### 3. Configure Health Checks

Railway will automatically use the `/healthz` endpoint for health checks.

Configure in service settings:
- Health Check Path: `/healthz`
- Health Check Timeout: 10 seconds

### 4. Configure Networking

1. Go to Settings > Networking
2. Enable "Public Networking"
3. Generate a domain or configure custom domain

### 5. Deploy

Railway will automatically build and deploy when you push to the configured branch.

Monitor the deployment in the Railway dashboard:
- Build logs show Docker build progress
- Deploy logs show application startup

## Dockerfile Context

The Dockerfile is designed for monorepo deployment:

```dockerfile
# Built from monorepo root
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/github-stats-api/
COPY ${APP_DIR}/src ./services/github-stats-api/src
COPY libs ./libs
```

This copies:
- The github-stats-api service
- Shared libs (`platform_core`)

## Usage

Once deployed, embed cards in GitHub README files:

**Stats Card:**
```markdown
![GitHub Stats](https://your-app.railway.app/api?username=your-username&theme=dracula)
```

**Languages Card:**
```markdown
![Top Languages](https://your-app.railway.app/api/top-langs?username=your-username&layout=compact)
```

## Monitoring

### Logs

View logs in Railway dashboard or CLI:

```bash
railway logs
```

Logs are structured JSON format for easy parsing.

### Health Checks

Monitor health endpoints:

```bash
# Liveness
curl https://your-app.railway.app/healthz

# Readiness
curl https://your-app.railway.app/readyz
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

1. Verify `GITHUB_TOKEN` is set and has correct scope
2. Check that token has not expired
3. Review application logs for errors

### Rate Limiting

If you hit GitHub API rate limits:
1. Increase `CACHE_TTL_SECONDS` to reduce API calls
2. Consider using a GitHub App token for higher limits
3. Monitor the 5,000 points/hour GraphQL API limit

### Connection Refused

1. Ensure `PORT` is not set (Railway provides it)
2. Verify health check is passing
3. Check networking settings

## Cost Optimization

- Use Railway's free tier for development
- Monitor usage in Railway dashboard
- Set resource limits to control costs
- Increase cache TTL to reduce compute usage

## Security Considerations

1. Use Railway's secrets management for `GITHUB_TOKEN`
2. HTTPS is automatic on Railway
3. Monitor logs for unusual request patterns
4. Rotate GitHub tokens periodically
