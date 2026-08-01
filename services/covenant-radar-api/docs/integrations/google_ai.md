# Google AI (Gemini) Integration

LLM-powered alert text generation for human-readable risk summaries.

---

## Overview

The Google AI integration provides:

- **Alert Summaries**: Human-readable text summaries for covenant breaches and high-risk predictions
- **GeminiClient**: Typed wrapper around the google-genai SDK
- **AlertContext**: Structured input with deal info, risk tier, and evaluation status
- **Usage Metrics**: Token counting and latency tracking for cost monitoring

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google AI API key (required for production) |

### Example Configuration

**Local development (with API key):**
```bash
export GEMINI_API_KEY=your-api-key-here
```

**Production deployment:**
```bash
export GEMINI_API_KEY=your-production-api-key
```

---

## Usage

### Creating a Client

```python
from covenant_radar_api.integrations.google_ai import (
    create_gemini_client,
    GeminiConfig,
)

config: GeminiConfig = {
    "api_key": os.environ["GEMINI_API_KEY"],
    "model": "gemini-2.5-flash",
}
client = create_gemini_client(config)
```

### Generating Alert Summaries

```python
from covenant_radar_api.integrations.google_ai import make_alert_context

context = make_alert_context(
    deal_id="deal-001",
    deal_name="Acme Corp Loan",
    borrower_name="Acme Corporation",
    sector="Technology",
    risk_probability=0.85,
    risk_tier="CRITICAL",
    evaluation_status="BREACH",
    breaches_count=2,
    covenants_evaluated=5,
    period_start="2024-01-01",
    period_end="2024-03-31",
)

response = client.generate_alert_summary(context)
summary = response["summary"]
input_tokens = response["input_tokens"]
output_tokens = response["output_tokens"]
latency_ms = response["latency_ms"]
```

### Raw Text Generation

```python
# For custom prompts
text = client.generate_text("Explain covenant compliance in one sentence.")
```

---

## Schemas

### AlertContext

Input context for alert generation.

| Field | Type | Description |
|-------|------|-------------|
| `deal_id` | str | Unique deal identifier |
| `deal_name` | str | Human-readable deal name |
| `borrower_name` | str | Borrower company name |
| `sector` | str | Industry sector |
| `risk_probability` | float | ML-predicted risk (0.0-1.0) |
| `risk_tier` | Literal | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `evaluation_status` | Literal | `OK`, `BREACH`, `WARNING` |
| `breaches_count` | int | Number of covenant breaches |
| `covenants_evaluated` | int | Total covenants evaluated |
| `period_start` | str | Period start date (ISO format) |
| `period_end` | str | Period end date (ISO format) |

### GenerateAlertResponse

Response from alert generation.

| Field | Type | Description |
|-------|------|-------------|
| `summary` | str | Generated human-readable summary |
| `input_tokens` | int | Input tokens used |
| `output_tokens` | int | Output tokens generated |
| `model` | str | Model that generated the response |
| `latency_ms` | int | API call latency in milliseconds |

---

## Prompt Template

The default prompt template formats the alert context for a credit risk officer:

```
You are a financial risk analyst assistant. Generate a concise,
professional alert summary for the following situation.

Deal Information:
- Deal ID: {deal_id}
- Deal Name: {deal_name}
- Borrower: {borrower_name}
- Sector: {sector}
- Period: {period_start} to {period_end}

Risk Assessment:
- ML Risk Probability: {risk_probability:.1%}
- Risk Tier: {risk_tier}
- Evaluation Status: {evaluation_status}
- Covenants Evaluated: {covenants_evaluated}
- Breaches Detected: {breaches_count}

Write a single paragraph (2-3 sentences) summarizing this alert for a
credit risk officer. Focus on the key risk factors and recommended
immediate actions. Be specific about the severity and urgency.
```

---

## Testing

The integration uses Protocol-based dependency injection for testing:

```python
from covenant_radar_api.integrations.google_ai._test_hooks import (
    FakeGeminiClient,
    use_fake_gemini,
)

# In tests, inject the fake client
fake = use_fake_gemini()
fake.next_response = "Test alert summary"

# Client will now use the fake
client = create_gemini_client(config)
response = client.generate_alert_summary(context)
assert response["summary"] == "Test alert summary"
```

### FakeGeminiClient Features

| Property | Type | Description |
|----------|------|-------------|
| `next_response` | str | Response to return from generate_content |
| `next_token_count` | int | Token count to return |
| `should_fail` | bool | If True, raises GeminiError |
| `fail_message` | str | Error message when should_fail is True |
| `generate_calls` | list | History of generate_content calls |
| `count_calls` | list | History of count_tokens calls |

---

## Error Handling

### GeminiError

Raised when the Gemini API fails or returns an empty response:

```python
from covenant_radar_api.integrations.google_ai._test_hooks import GeminiError

try:
    response = client.generate_alert_summary(context)
except GeminiError as e:
    logger.error(f"Gemini API failed: {e}")
```

---

## Model Selection

The `model` field in GeminiConfig controls which Gemini model is used:

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Fast, low-cost generation (default) |
| `gemini-2.5-pro` | Higher quality, slower generation |

`gemini-2.0-flash` is no longer usable — Google retired it, and requests now
return `404 NOT_FOUND`. It was this service's default until 2026-07-31, which
meant every alert summary failed. Model names are worth re-checking against
`client.models.list()` rather than assumed stable.

---

## Cost Considerations

- **Token Usage**: Track `input_tokens` and `output_tokens` from responses
- **Model Selection**: Use `gemini-2.5-flash` for cost-effective alert generation
- **Caching**: Consider caching summaries for identical context to reduce API calls
- **Rate Limits**: Google AI has rate limits; implement backoff for high-volume usage
