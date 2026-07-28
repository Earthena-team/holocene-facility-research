import pytest
import fakeredis.aioredis as fakeredis

from shared.config import Settings
from shared.redis_queue import JobQueue


@pytest.fixture
def settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/15",
        daily_budget_usd=25.0,
        max_attempts=3,
        heartbeat_stale_seconds=1,
        dedupe_ttl_days=90,
    )


@pytest.fixture
async def queue(settings: Settings):
    client = fakeredis.FakeRedis(decode_responses=True)
    q = JobQueue(client, settings)
    yield q
    await client.flushdb()
    await client.aclose()
