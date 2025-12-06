# platform-discord

Typed Discord platform helpers: Redis event subscriber and embed utilities.

## Installation

```bash
poetry add platform-discord
```

Requires `discord.py` and `platform-workers` as peer dependencies.

## Event Subscriber

Subscribe to Redis channels and dispatch typed events to handlers:

```python
from platform_discord.subscriber import RedisEventSubscriber, MessageSource

# Define your event type
from typing import TypedDict

class MyEvent(TypedDict):
    type: str
    data: str

# Implement MessageSource protocol (wraps Redis pubsub)
class RedisSource(MessageSource):
    async def subscribe(self, channel: str) -> None: ...
    async def get(self) -> str | None: ...
    async def close(self) -> None: ...

# Create subscriber
subscriber = RedisEventSubscriber[MyEvent](
    channel="events:my-channel",
    source=redis_source,
    decode=lambda s: json.loads(s),  # str -> MyEvent | None
    handle=my_async_handler,          # async (MyEvent) -> None
)

# Run event loop
await subscriber.run()  # or run(limit=100) for testing
```

## Embed Helpers

Typed wrappers for discord.py embeds (avoids `Any` types):

```python
from platform_discord.embed_helpers import (
    create_embed,
    add_field,
    set_footer,
    get_field,
    get_footer_text,
    unwrap_embed,
)

# Create embed
embed = create_embed(
    title="Status Update",
    description="Job completed",
    color=0x57F287  # green
)

# Add fields
add_field(embed, name="Job ID", value="abc123", inline=True)
add_field(embed, name="Duration", value="5.2s", inline=True)
set_footer(embed, text="Request ID: xyz789")

# Read properties (typed)
field = get_field(embed, "Job ID")
if field:
    print(field["value"])  # "abc123"

# Pass to discord.py APIs
await interaction.followup.send(embed=unwrap_embed(embed))
```

### Embed Types

```python
from platform_discord.embed_helpers import (
    EmbedProto,       # Protocol for embed operations
    EmbedData,        # TypedDict for full embed
    EmbedFieldData,   # TypedDict: name, value, inline
    EmbedFooterData,  # TypedDict: text, icon_url
    EmbedAuthorData,  # TypedDict: name, icon_url, url
)
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- discord.py 2.3.0+
- platform-workers (peer dependency)
- 100% test coverage enforced

## Domain Kits

Typed, ready-to-use kits for common domains. Each kit provides strict types, embed builders, and optional runtime helpers:

- platform_discord.handwriting – digits training/progress embeds and runtime
- platform_discord.trainer – model-trainer progress embeds and runtime
- platform_discord.turkic – Turkic job progress embeds and runtime
- platform_discord.qr – QR result embeds
- platform_discord.transcript – transcript result embeds

Example: Turkic runtime

```python
from platform_core.job_events import JobEventV1
from platform_discord.turkic.runtime import new_runtime, on_started

rt = new_runtime()
act = on_started(rt, user_id=123, job_id="abc", queue="default")
if act["embed"] is not None:
    await user.send(embed=act["embed"])  # EmbedProto
```

Example: QR embed

```python
from platform_discord.qr.embeds import build_qr_result
from platform_discord.embed_helpers import unwrap_embed

embed = build_qr_result(dest_url="https://example.com", pixels=2048)
await interaction.followup.send(embed=unwrap_embed(embed))
```

Bot subscriber pattern

```python
from platform_core.job_events import JobEventV1, decode_job_event, default_events_channel
from platform_discord.subscriber import RedisEventSubscriber
from platform_discord.message_source import RedisPubSubSource
from platform_discord.turkic.runtime import new_runtime, on_started, on_progress, on_completed, on_failed

source = RedisPubSubSource(redis_url)
rt = new_runtime()

async def handle(ev: JobEventV1) -> None:
    if ev["type"] == "turkic.job.started.v1":
        act = on_started(rt, user_id=None, job_id=ev["job_id"], queue=ev["queue"])
    elif ev["type"] == "turkic.job.progress.v1":
        act = on_progress(rt, user_id=None, job_id=ev["job_id"], progress=ev["progress"], message=ev.get("message"))
    elif ev["type"] == "turkic.job.completed.v1":
        act = on_completed(rt, user_id=None, job_id=ev["job_id"], result_id=ev["result_id"], result_bytes=ev["result_bytes"])
    else:
        act = on_failed(rt, user_id=None, job_id=ev["job_id"], error_kind=ev.get("error_kind", "system"), message=ev.get("message", "unknown event"), status="failed")
    if act["embed"]:
        await user.send(embed=act["embed"])  # your user lookup

sub = RedisEventSubscriber[JobEventV1](
    channel=default_events_channel("turkic"),
    source=source,
    decode=decode_job_event,
    handle=handle,
)
await sub.run()
```
