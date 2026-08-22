"""The dashboard page template and static chart constants."""

from __future__ import annotations

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Covenant Radar Dashboard</title>
    <script src="/dashboard/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e8e8e8;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 {
            font-size: 1.5rem;
            color: #00d9ff;
        }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-online { background: #10b981; color: #fff; }
        .status-offline { background: #ef4444; color: #fff; }
        .main-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
        }
        .card {
            background: rgba(15, 52, 96, 0.5);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(0, 217, 255, 0.2);
        }
        .card h2 {
            font-size: 1rem;
            color: #00d9ff;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .metric-box {
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00d9ff;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #888;
            margin-top: 0.25rem;
        }
        .risk-low { color: #10b981; }
        .risk-medium { color: #f59e0b; }
        .risk-high { color: #f97316; }
        .risk-critical { color: #ef4444; }
        .chart-container {
            position: relative;
            height: 250px;
        }
        .predictions-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: rgba(0,0,0,0.2);
            margin-bottom: 0.5rem;
            border-radius: 6px;
            border-left: 3px solid transparent;
        }
        .prediction-item.critical { border-left-color: #ef4444; }
        .prediction-item.high { border-left-color: #f97316; }
        .prediction-item.medium { border-left-color: #f59e0b; }
        .prediction-item.low { border-left-color: #10b981; }
        .prediction-deal { font-weight: 600; font-size: 0.9rem; }
        .prediction-time { font-size: 0.75rem; color: #888; }
        .prediction-prob {
            font-size: 1.25rem;
            font-weight: 700;
        }
        .jobs-list {
            max-height: 250px;
            overflow-y: auto;
        }
        .job-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: rgba(0,0,0,0.2);
            margin-bottom: 0.5rem;
            border-radius: 6px;
        }
        .job-status {
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .job-queued { background: #6366f1; }
        .job-started { background: #f59e0b; }
        .job-finished { background: #10b981; }
        .job-failed { background: #ef4444; }
        .model-info {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .model-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .model-label { color: #888; }
        .model-value { color: #fff; font-weight: 600; }
        .model-path { font-size: 0.8rem; }
        .job-id { font-weight: 600; font-size: 0.9rem; }
        .refresh-btn {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 0.5rem 1rem;
            margin-left: 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        .refresh-btn:hover { opacity: 0.8; }
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading { animation: pulse 1.5s infinite; }
    </style>
</head>
<body>
    <header class="header">
        <h1>Covenant Radar Dashboard</h1>
        <div>
            <span id="status-badge" class="status-badge status-offline">Checking...</span>
            <button class="refresh-btn" onclick="refreshWithPredictions()">Refresh</button>
        </div>
    </header>

    <main class="main-grid">
        <div class="card">
            <h2>Risk Distribution</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-value risk-critical" id="count-critical">0</div>
                    <div class="metric-label">CRITICAL</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value risk-high" id="count-high">0</div>
                    <div class="metric-label">HIGH</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value risk-medium" id="count-medium">0</div>
                    <div class="metric-label">MEDIUM</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value risk-low" id="count-low">0</div>
                    <div class="metric-label">LOW</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Active Model</h2>
            <div class="model-info" id="model-info">
                <div class="loading">Loading model info...</div>
            </div>
        </div>

        <div class="card">
            <h2>Probability Trend</h2>
            <div class="chart-container">
                <canvas id="trend-chart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Recent Predictions</h2>
            <div class="predictions-list" id="predictions-list">
                <div class="empty-state">No predictions yet</div>
            </div>
        </div>

        <div class="card">
            <h2>Background Jobs</h2>
            <div class="jobs-list" id="jobs-list">
                <div class="empty-state">No active jobs</div>
            </div>
        </div>

        <div class="card">
            <h2>Risk Distribution Chart</h2>
            <div class="chart-container">
                <canvas id="risk-chart"></canvas>
            </div>
        </div>
    </main>

    <script>
        // Escape before interpolating any API-derived string into innerHTML.
        // Deal names and borrowers are free text written through POST /deals,
        // so unescaped interpolation both breaks rendering on characters like
        // & and < and lets stored markup execute in an operator's browser.
        function escapeHtml(value) {
            const entities = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            return String(value === null || value === undefined ? '' : value)
                .replace(/[&<>"']/g, function (c) { return entities[c]; });
        }

        // State
        let predictions = [];
        let trendChart = null;
        let riskChart = null;
        let trackedJobs = [];
        let deals = [];

        // Initialize charts
        function initCharts() {
            const trendCtx = document.getElementById('trend-chart').getContext('2d');
            trendChart = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Breach Probability',
                        data: [],
                        borderColor: '#00d9ff',
                        backgroundColor: 'rgba(0, 217, 255, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            min: 0,
                            max: 1,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#888' }
                        },
                        x: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#888', maxRotation: 45 }
                        }
                    }
                }
            });

            const riskCtx = document.getElementById('risk-chart').getContext('2d');
            riskChart = new Chart(riskCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#ef4444', '#f97316', '#f59e0b', '#10b981'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#888' }
                        }
                    }
                }
            });
        }

        // Check API health
        async function checkHealth() {
            try {
                const response = await fetch('/healthz');
                const badge = document.getElementById('status-badge');
                if (response.ok) {
                    badge.textContent = 'Online';
                    badge.className = 'status-badge status-online';
                } else {
                    badge.textContent = 'Offline';
                    badge.className = 'status-badge status-offline';
                }
            } catch (e) {
                document.getElementById('status-badge').textContent = 'Offline';
                document.getElementById('status-badge').className = 'status-badge status-offline';
            }
        }

        // Fetch model info
        async function fetchModelInfo() {
            try {
                const response = await fetch('/ml/models/active');
                const data = await response.json();
                const container = document.getElementById('model-info');
                const statusColor = data.is_loaded ? '#10b981' : '#ef4444';
                const statusText = data.is_loaded ? 'Loaded' : 'Not Loaded';
                const pathVal = escapeHtml(data.model_path || 'N/A');
                const modelId = escapeHtml(data.model_id || 'N/A');
                container.innerHTML = `
                    <div class="model-row">
                        <span class="model-label">Model ID</span>
                        <span class="model-value">${modelId}</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Path</span>
                        <span class="model-value model-path">${pathVal}</span>
                    </div>
                    <div class="model-row">
                        <span class="model-label">Status</span>
                        <span class="model-value" style="color: ${statusColor}">
                            ${statusText}
                        </span>
                    </div>
                `;
            } catch (e) {
                const errMsg = 'Failed to load model info';
                document.getElementById('model-info').innerHTML =
                    '<div class="empty-state">' + errMsg + '</div>';
            }
        }

        // Update risk counts
        function updateRiskCounts() {
            const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
            predictions.forEach(p => counts[p.risk_tier]++);

            document.getElementById('count-critical').textContent = counts.CRITICAL;
            document.getElementById('count-high').textContent = counts.HIGH;
            document.getElementById('count-medium').textContent = counts.MEDIUM;
            document.getElementById('count-low').textContent = counts.LOW;

            // Update doughnut chart
            riskChart.data.datasets[0].data = [
                counts.CRITICAL, counts.HIGH, counts.MEDIUM, counts.LOW
            ];
            riskChart.update();
        }

        // Add prediction (simulated or from API)
        function addPrediction(prediction) {
            predictions.unshift(prediction);
            if (predictions.length > 50) predictions.pop();

            updateRiskCounts();
            renderPredictions();
            updateTrendChart();
        }

        // Render predictions list
        function renderPredictions() {
            const container = document.getElementById('predictions-list');
            if (predictions.length === 0) {
                container.innerHTML = '<div class="empty-state">No predictions yet</div>';
                return;
            }

            container.innerHTML = predictions.slice(0, 20).map(p => {
                const tier = escapeHtml(p.risk_tier.toLowerCase());
                const pct = (p.probability * 100).toFixed(1);
                const displayName = escapeHtml(p.deal_name || p.deal_id.slice(0, 8) + '...');
                const displayBorrower = escapeHtml(p.borrower || '');
                return `
                <div class="prediction-item ${tier}">
                    <div>
                        <div class="prediction-deal">${displayName}</div>
                        <div class="prediction-time">${displayBorrower}</div>
                    </div>
                    <div class="prediction-prob risk-${tier}">${pct}%</div>
                </div>
            `}).join('');
        }

        // Update trend chart
        function updateTrendChart() {
            const recent = predictions.slice(0, 20).reverse();
            trendChart.data.labels = recent.map((_, i) => `#${i + 1}`);
            trendChart.data.datasets[0].data = recent.map(p => p.probability);
            trendChart.update();
        }

        // Fetch job status
        async function fetchJobStatus(jobId) {
            try {
                // URL context, not HTML: encode so a job id containing / or ?
                // cannot reshape the request path.
                const response = await fetch(`/ml/jobs/${encodeURIComponent(jobId)}`);
                return await response.json();
            } catch (e) {
                return { job_id: jobId, status: 'unknown' };
            }
        }

        // Render jobs list
        async function renderJobs() {
            const container = document.getElementById('jobs-list');
            if (trackedJobs.length === 0) {
                container.innerHTML = '<div class="empty-state">No tracked jobs</div>';
                return;
            }

            const jobStatuses = await Promise.all(trackedJobs.map(fetchJobStatus));
            container.innerHTML = jobStatuses.map(job => {
                const jobId = escapeHtml(job.job_id.slice(0, 8));
                const status = escapeHtml(job.status);
                return `
                <div class="job-item">
                    <div>
                        <div class="job-id">${jobId}...</div>
                    </div>
                    <span class="job-status job-${status}">${status}</span>
                </div>
            `}).join('');
        }

        // Fetch all deals from the API
        async function fetchDeals() {
            try {
                const response = await fetch('/deals');
                if (!response.ok) {
                    console.error('Failed to fetch deals:', response.status);
                    return [];
                }
                const data = await response.json();
                deals = data;
                return data;
            } catch (e) {
                console.error('Error fetching deals:', e);
                return [];
            }
        }

        // Predict breach risk for a single deal
        async function predictDeal(dealId) {
            try {
                const response = await fetch('/ml/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ deal_id: dealId })
                });
                if (!response.ok) {
                    console.error('Failed to predict deal:', dealId, response.status);
                    return null;
                }
                return await response.json();
            } catch (e) {
                console.error('Error predicting deal:', dealId, e);
                return null;
            }
        }

        // Fetch all deals and predict each one
        async function fetchAndPredictAllDeals() {
            const dealList = await fetchDeals();
            if (dealList.length === 0) {
                console.log('No deals to predict');
                return;
            }

            // Clear existing predictions
            predictions = [];

            // Predict each deal sequentially to avoid overwhelming the API
            for (const deal of dealList) {
                const dealId = deal.id.value;
                const result = await predictDeal(dealId);
                if (result) {
                    addPrediction({
                        deal_id: dealId,
                        deal_name: deal.name,
                        borrower: deal.borrower,
                        probability: result.probability,
                        risk_tier: result.risk_tier,
                        timestamp: new Date().toLocaleTimeString()
                    });
                }
            }
        }

        // Simulate streaming predictions for demo
        function simulatePrediction() {
            const tiers = ['LOW', 'LOW', 'LOW', 'MEDIUM', 'MEDIUM', 'HIGH', 'CRITICAL'];
            const tier = tiers[Math.floor(Math.random() * tiers.length)];
            let prob;
            switch (tier) {
                case 'LOW': prob = Math.random() * 0.25; break;
                case 'MEDIUM': prob = 0.25 + Math.random() * 0.25; break;
                case 'HIGH': prob = 0.5 + Math.random() * 0.25; break;
                case 'CRITICAL': prob = 0.75 + Math.random() * 0.25; break;
            }

            addPrediction({
                deal_id: crypto.randomUUID(),
                probability: prob,
                risk_tier: tier,
                timestamp: new Date().toLocaleTimeString()
            });
        }

        // Refresh all data
        async function refreshAll() {
            await checkHealth();
            await fetchModelInfo();
            await renderJobs();
        }

        // Full refresh including predictions
        async function refreshWithPredictions() {
            await refreshAll();
            await fetchAndPredictAllDeals();
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            refreshWithPredictions();

            // Refresh health and jobs every 30 seconds
            setInterval(refreshAll, 30000);
        });

        // Add a job to track (can be called from console)
        window.trackJob = function(jobId) {
            trackedJobs.push(jobId);
            renderJobs();
        };
    </script>
</body>
</html>
"""
