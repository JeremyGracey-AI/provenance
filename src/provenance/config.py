"""Runtime settings, read from `PROVENANCE_*` environment variables (or a local `.env`).

The same `Settings` object configures the offline demo (no keys needed) and the real
deployment (Cohere + Claude). Keys are optional here so the offline fakes and CI never
require secrets; the real backends assert their own keys at construction.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from provenance import records


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROVENANCE_", env_file=".env", extra="ignore")

    # Secrets / model selection.
    anthropic_api_key: str | None = None  # falls back to the SDK's ANTHROPIC_API_KEY if unset
    cohere_api_key: str | None = None
    vlm_model: str = "claude-sonnet-4-6"  # current vision-capable Claude; see Anthropic's model list

    @field_validator("vlm_model")
    @classmethod
    def _vlm_model_must_be_recordable(cls, value: str) -> str:
        """The operator's door for `model` — WEEK-5 Day 1 item 3.

        The defect, open and named in three docstrings since 2026-08-02:
        `PROVENANCE_VLM_MODEL=" "` was HTTP 200, `build_record` copied it into EVERY record, and
        `--verify` then failed each one with `field=model — empty`. Never caller-reachable —
        it needs whoever controls the deployment's environment — but it is the last known way
        an answer that returns 200 leaves behind a record its own verifier rejects.

        REJECT AT `Settings`, rather than defaulting on blank. Both were live options and the
        choice is arguable, so here is the argument:

          * Defaulting SUBSTITUTES a model the operator did not choose. Every record would then
            name a model that was never configured — a lie in the audit trail, and the same
            silent-substitution shape as stripping a NUL out of model output rather than
            refusing it (`protocols.MalformedModelOutput`). A record store whose `model` column
            cannot be trusted is worse than one that is missing rows for a reason someone can
            see.
          * Rejecting fails at PROCESS START. `api/index.py:23` calls `load_settings()` at
            import, so a blank value takes the deployment down at deploy time with a
            `ValidationError` naming `vlm_model`, in the platform log, before a single caller
            is served a record that cannot be verified. The cost is real and is stated rather
            than hidden: this is a correctness-over-availability choice. It is the right one
            here because the deliverable IS the verifiable record, and because the trigger is
            an operator deliberately setting the variable to whitespace — an unset variable
            keeps the default above and nothing changes.
          * It keeps `is_present` doing one job. Its docstring's own asymmetry — "this decides
            *whether* to keep a value, never *what* the value is" — would break the moment
            config started rewriting one.

        Both predicates, both SHARED, both resolved as module attributes at call time, exactly
        as the two doors in `api/app.py` and the guard in `protocols.py` resolve them: these
        are the same two rules `_schema_violations` and `_nul_violations` apply to `model`, so
        the door and the verifier agree by construction rather than by coincidence. Nothing
        else is checked — not a charset, not a length, not membership in a model list. A
        narrower rule would refuse model ids Anthropic has not published yet; a wider one would
        be a restriction with no driver behind it.
        """
        if not records.is_present(value):
            raise ValueError(
                "must contain at least one non-whitespace character — a blank model is copied "
                "into every record and `--verify` rejects each one with `field=model — empty`"
            )
        if not records.is_nul_free(value):
            raise ValueError(
                "must not contain U+0000 (NUL) — the record store cannot represent it"
            )
        return value  # verbatim: rejects, never rewrites (no strip), like every other door here

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
