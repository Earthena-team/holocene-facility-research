from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_schema: str = "public"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    geocoding_api_key: str = ""

    daily_budget_usd: float = 25.0
    refresh_days: int = 90
    max_attempts: int = 3
    worker_concurrency: int = 2
    thinking_budget: int = 1024
    max_output_tokens: int = 12288
    min_facilities_before_retry: int = 3

    price_in_per_m: float = 1.50
    price_out_per_m: float = 9.00
    price_per_1k_search: float = 14.00
    price_per_1k_geocode: float = 5.00

    dedupe_ttl_days: int = 90
    worker_health_port: int = 8080

    # Redis TTLs / operational constants
    job_ttl_seconds: int = 7 * 24 * 3600
    halt_ttl_seconds: int = 24 * 3600
    geocode_ttl_seconds: int = 180 * 24 * 3600
    budget_ttl_seconds: int = 48 * 3600
    metrics_ttl_seconds: int = 48 * 3600
    heartbeat_stale_seconds: int = 10 * 60
    heartbeat_interval_seconds: int = 15
    max_facilities: int = 40
    geocode_concurrency: int = 5
    geocode_distance_km_threshold: float = 50.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
