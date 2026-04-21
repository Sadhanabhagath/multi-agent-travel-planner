"""
config/settings.py
------------------
Centralized configuration using pydantic-settings.
All settings are read from environment variables or .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm_model: str = "gpt-4o-mini"
    orchestrator_llm_model: str = "gpt-4o"

    # ── Travel APIs ───────────────────────────────────────────
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    amadeus_environment: str = "test"
    google_places_api_key: str = ""
    exchange_rates_api_key: str = ""

    # ── Agent Behavior ────────────────────────────────────────
    max_flight_results: int = 5
    max_hotel_results: int = 5
    max_activity_results: int = 10
    budget_buffer_percent: float = 10.0
    max_agent_iterations: int = 10
    enable_human_in_loop: bool = True

    # ── Caching ───────────────────────────────────────────────
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    redis_url: str = ""

    # ── Observability ─────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "travel-planner"
    enable_tracing: bool = False

    # ── App ───────────────────────────────────────────────────
    log_level: str = "INFO"
    app_env: str = "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def has_amadeus(self) -> bool:
        return bool(self.amadeus_client_id and self.amadeus_client_secret)

    @property
    def has_google_places(self) -> bool:
        return bool(self.google_places_api_key)

    @property
    def llm_provider(self) -> str:
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return "none"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
