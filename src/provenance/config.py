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

    # Decision records. Which sink runs is CONFIG, not a branch in the caller —
    # `records.sink_from_settings(settings)` reads exactly these fields:
    #   url + key set -> HttpSink     (durable, survives the request)
    #   dir set       -> JsonlSink    (local file, one per day)
    #   neither       -> None         (no records at all, by design and visibly so)
    records_url: str | None = None  # REST endpoint, e.g. https://<ref>.supabase.co/rest/v1/answer_records
    records_key: str | None = None  # API key for that endpoint; server-side only, never shipped to a browser
    records_dir: str | None = None  # local JSONL directory (JsonlSink) when no URL is configured
    records_timeout_s: float = 2.0  # hard cap on the record POST — an answer never waits on the store

    # Salt for the client-address hash written into records (see `records.hash_client`).
    # Documented default: UNSET means the API mints a RANDOM per-process salt at
    # `create_app` time. Consequence, stated so nobody discovers it in a dashboard: hashes
    # are then comparable only within one running instance and change on every restart or
    # new serverless instance. That is the privacy-conservative default; set this to a
    # stable secret only if you actually want cross-instance correlation, and treat it as a
    # secret when you do — a KNOWN salt makes the 2^32 IPv4 space brute-forceable.
    client_hash_salt: str | None = None


def load_settings() -> Settings:
    return Settings()
