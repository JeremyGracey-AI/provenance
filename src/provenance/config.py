"""Runtime settings, read from `PROVENANCE_*` environment variables (or a local `.env`).

The same `Settings` object configures the offline demo (no keys needed) and the real
deployment (Cohere + Claude). Keys are optional here so the offline fakes and CI never
require secrets; the real backends assert their own keys at construction.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROVENANCE_", env_file=".env", extra="ignore")

    # Secrets / model selection.
    anthropic_api_key: str | None = None  # falls back to the SDK's ANTHROPIC_API_KEY if unset
    cohere_api_key: str | None = None
    vlm_model: str = "claude-sonnet-4-6"  # current vision-capable Claude; see Anthropic's model list

    # Corpus image host: HF resolve base, e.g.
    # https://huggingface.co/datasets/<user>/<repo>/resolve/main/pages
    pages_base_url: str | None = None

    # Pipeline knobs.
    top_k: int = 5  # pages retrieved per query
    max_repairs: int = 1  # re-answer attempts when claims come back unsupported
    answer_max_tokens: int = 2048  # headroom for a detailed answer + several cited claims in one tool call
    judge_max_tokens: int = 512


def load_settings() -> Settings:
    return Settings()
