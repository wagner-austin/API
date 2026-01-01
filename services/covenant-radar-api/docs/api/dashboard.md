# Dashboard Endpoint

Real-time monitoring dashboard UI for covenant breach risk visualization.

---

## GET /dashboard

Serves an HTML page with a single-page monitoring dashboard for real-time breach risk visualization.

**Response Content-Type:** `text/html`

**Response:** HTML page with embedded JavaScript that provides:

- **Service Status**: Online/Offline indicator (checks `/healthz`)
- **Risk Distribution**: Counts by risk tier (Critical, High, Medium, Low)
- **Active Model Info**: Displays model ID, path, and loaded status (fetches from `/ml/models/active`)
- **Probability Trend Chart**: Line chart of recent prediction probabilities
- **Recent Predictions**: List of recent predictions with deal ID, timestamp, and probability
- **Background Jobs**: List of tracked training/optimization jobs (fetched from `/ml/jobs/{job_id}`)
- **Risk Distribution Chart**: Doughnut chart visualization of risk tiers

### Features

- **Auto-refresh**: Health and job status refresh every 10 seconds
- **Manual refresh**: Click "Refresh" button to update all data
- **Job tracking**: Call `trackJob(jobId)` in browser console to add jobs to the monitoring list

### Usage

Open the dashboard in a browser:

```
http://localhost:8007/dashboard
```

Or in production:

```
https://covenant-radar-api-production.up.railway.app/dashboard
```

### Screenshot

The dashboard displays:
- Header with service name and online/offline status badge
- Grid layout with cards for metrics, charts, and lists
- Dark theme with cyan (#00d9ff) accent colors
