# Configuration Reference

Every environment variable the service reads, with its default. Values are
parsed by `platform_core.config.load_covenant_radar_settings` (API + RQ worker)
and `streaming.config.load_streaming_config` (streaming).

## API and RQ Worker

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_ENV` | string | `dev` | Application environment (`dev` or `prod`) |
| `DATABASE_URL` | string | - | PostgreSQL connection URL (**required** — the service refuses to start without it, rather than silently attaching to a local socket) |
| `REDIS_URL` or `REDIS__URL` | string | `redis://redis:6379/0` | Redis connection URL |
| `REDIS__ENABLED` | bool | `true` | Enable Redis |
| `RQ__QUEUE_NAME` | string | `covenant` | RQ queue name |
| `RQ__JOB_TIMEOUT_SEC` | int | `3600` | Job timeout in seconds |
| `RQ__RESULT_TTL_SEC` | int | `86400` | Result TTL in seconds |
| `RQ__FAILURE_TTL_SEC` | int | `604800` | Failure TTL in seconds |
| `APP__DATA_ROOT` | string | `/data` | Data root directory |
| `APP__MODELS_ROOT` | string | `/data/models` | Models directory; every request-supplied `model_path` must resolve inside it |
| `APP__LOGS_ROOT` | string | `/data/logs` | Logs directory |
| `APP__ML_BACKEND` | string | `xgboost` | Inference backend (`xgboost`, `mlp`, `lstm`, `lightgbm`); selects which active model path below is used |
| `APP__ACTIVE_MODEL_PATH_XGB` | string | `/data/models/active_xgb.ubj` | Active model path when the backend is `xgboost` |
| `APP__ACTIVE_MODEL_PATH_MLP` | string | `/data/models/active_mlp.pt` | Active model path for the other backends |
| `LOGGING__LEVEL` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `DATA_BANK_API_URL` | string | - | URL for data-bank-api |
| `DATA_BANK_API_KEY` | string | - | API key for data-bank-api uploads/downloads |
| `DATA_BANK_MODEL_FILE_ID` | string | - | Model file_id (SHA256) to download from data-bank at startup |
| `GEMINI_API_KEY` | string | - | Google AI API key for Gemini integration |

There is no `APP__ACTIVE_MODEL_PATH`. The active path is derived from
`APP__ML_BACKEND` plus the two backend-specific paths above.

## Datadog

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATADOG__ENABLED` | bool | `false` | Enable Datadog integration |
| `DATADOG__SERVICE` | string | `covenant-radar-api` | Service name for traces |
| `DATADOG__ENV` | string | `dev` | Environment (`dev`, `staging`, `production`) |
| `DATADOG__VERSION` | string | `0.0.0` | Service version |
| `DATADOG__AGENT_HOST` | string | `localhost` | Datadog agent host |
| `DATADOG__DOGSTATSD_PORT` | int | `8125` | DogStatsD UDP port |
| `DATADOG__TRACE_ENABLED` | bool | `true` | Enable APM tracing |

See [integrations/datadog.md](./integrations/datadog.md) for the metric catalog.

## Kafka Streaming

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `STREAMING__ENABLED` | bool | `false` | Enable Kafka streaming |
| `CONFLUENT__BOOTSTRAP_SERVERS` | string | - | Confluent Cloud bootstrap servers |
| `CONFLUENT__API_KEY` / `CONFLUENT__API_SECRET` | string | - | SASL credentials (unused under `PLAINTEXT`) |
| `CONFLUENT__SECURITY_PROTOCOL` | string | `SASL_SSL` | `SASL_SSL` for Confluent Cloud, `PLAINTEXT` for a local or CI broker |
| `CONFLUENT__SCHEMA_REGISTRY_URL` | string | - | Schema Registry URL; setting it enables the registry config |
| `CONFLUENT__SCHEMA_REGISTRY_API_KEY` / `_API_SECRET` | string | - | Schema Registry credentials |
| `KAFKA__TOPIC_MEASUREMENTS` | string | `covenant.measurements.v1` | Input topic |
| `KAFKA__TOPIC_PREDICTIONS` | string | `covenant.predictions.v1` | Predictions output topic |
| `KAFKA__TOPIC_ALERTS` | string | `covenant.alerts.v1` | Alerts output topic |
| `KAFKA__TOPIC_DLQ` | string | `covenant.dlq.v1` | Dead-letter topic for undecodable payloads |
| `KAFKA__CONSUMER_GROUP_ID` | string | `covenant-radar-api` | Consumer group ID |
| `KAFKA__AUTO_OFFSET_RESET` | string | `earliest` | Offset reset policy (`earliest` or `latest`) |
| `KAFKA__ENABLE_AUTO_COMMIT` | bool | `false` | Auto-commit offsets |
| `KAFKA__FETCH_MIN_BYTES` | int | `1` | Minimum fetch bytes |
| `KAFKA__SESSION_TIMEOUT_MS` | int | `45000` | Session timeout |
| `KAFKA__HEARTBEAT_INTERVAL_MS` | int | `15000` | Heartbeat interval |
| `KAFKA__PRODUCER_ACKS` | string | `all` | Producer acknowledgment (`all`, `0`, `1`) |
| `KAFKA__PRODUCER_RETRIES` | int | `3` | Producer retries |
| `KAFKA__PRODUCER_LINGER_MS` | int | `5` | Producer linger time |
| `KAFKA__PRODUCER_BATCH_SIZE` | int | `16384` | Producer batch size |
| `KAFKA__COMPRESSION_TYPE` | string | `gzip` | `none`, `gzip`, `snappy`, `lz4`, or `zstd` |

## Streaming Worker

Read by `covenant-streaming-worker` (the `streaming` Docker target) only. The
worker exits non-zero unless `STREAMING__ENABLED` is true.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `STREAMING__DOMAIN` | string | `weather` | Which registered domain to run (`weather` or `esports`) |
| `STREAMING__POLL_TIMEOUT_SECONDS` | float | `1.0` | Consumer poll timeout |
| `MODEL_PATH` | string | - | Path to the saved model file (required) |
| `MODEL_VERSION` | string | `v1.0.0` | Version string reported on every prediction event |
| `GEMINI_API_KEY` | string | - | Required — alert summary generation |
| `GEMINI_MODEL` | string | `gemini-2.5-flash` | Gemini model used for alert summaries |
| `WEATHER__STATE_PATH` | string | - | Fitted temporal feature state, JSON (required for `weather`) |
| `WEATHER__STATION_MAP_PATH` | string | - | station_id to location index, JSON (required for `weather`) |
| `WEATHER__ALERT_THRESHOLD` | float | `0.80` | Probability at or above which a weather alert fires |
| `ESPORTS__ALERT_THRESHOLD` | float | `0.85` | Win probability at or above which an esports alert fires |

## Example .env

```bash
APP_ENV=dev
DATABASE_URL=postgresql://covenant:covenant@postgres:5432/covenant
REDIS_URL=redis://redis:6379/0
RQ__QUEUE_NAME=covenant
APP__ML_BACKEND=xgboost
APP__ACTIVE_MODEL_PATH_XGB=/data/models/active_xgb.ubj
LOGGING__LEVEL=INFO

# Optional, all disabled/empty by default
DATADOG__ENABLED=false
STREAMING__ENABLED=false
CONFLUENT__BOOTSTRAP_SERVERS=pkc-xxxxx.us-east-1.aws.confluent.cloud:9092
CONFLUENT__API_KEY=your-api-key
CONFLUENT__API_SECRET=your-api-secret
GEMINI_API_KEY=your-gemini-api-key

# Streaming worker only (covenant-streaming-worker)
STREAMING__DOMAIN=weather
MODEL_PATH=/data/models/weather.ubj
WEATHER__STATE_PATH=/data/models/weather_state.json
WEATHER__STATION_MAP_PATH=/data/models/weather_stations.json

# Data Bank (optional, for centralized model storage)
DATA_BANK_API_URL=http://data-bank-api.railway.internal:8080
DATA_BANK_API_KEY=your-data-bank-api-key
DATA_BANK_MODEL_FILE_ID=2d768b7fdb6b4919f6ddf1ace2bfef23433ec19f2ae2f7c3c454d9f9e23cfede
```
