-- Provenance decision records — proposed durable table (Supabase / PostgREST).
--
-- STATUS: PROPOSAL. Nothing here has been executed against any remote database. Creating
-- the table, minting the key, and setting the Vercel env vars are the human's call
-- (WEEK-4.md Workstream A Day 2); this file exists so that call is one paste, not a design
-- session.
--
-- Column names are EXACTLY the JSON keys that `src/provenance/records.py:build_record`
-- emits, because PostgREST maps a posted JSON object to columns by name. Renaming a column
-- here silently drops that field on insert.
--
-- Apply in the Supabase SQL editor, then set on the Vercel API project:
--   PROVENANCE_RECORDS_URL = https://<project-ref>.supabase.co/rest/v1/answer_records
--   PROVENANCE_RECORDS_KEY = <service_role key>
--   PROVENANCE_CLIENT_HASH_SALT = <a long random secret>   (optional; see below)

create table if not exists public.answer_records (
    -- run_id is the record's own uuid4 hex from build_record. Making it the primary key
    -- enforces verifier check (d) — run_id unique across the set — in the STORE, not just
    -- in `--verify`, and makes a retried POST idempotent instead of duplicating a run.
    run_id            text primary key,

    record_version    integer          not null,
    -- Quoted because `timestamp` is a type name in Postgres. It must keep this name: it is
    -- the JSON key, and JsonlSink's day-file naming derives from it.
    "timestamp"       timestamptz      not null,
    question          text             not null,
    model             text             not null,
    k                 integer          not null check (k >= 1),
    retrieved         jsonb            not null,   -- [{id, score}] in retrieval order
    claims            jsonb            not null,   -- [{text, citations, verdict, verify_fallback}]
    confidence        double precision not null check (confidence >= 0 and confidence <= 1),
    repairs           integer          not null check (repairs >= 0),
    trace             jsonb            not null,   -- [{name, duration_ms, detail, started_at}]

    -- Requester identity. NULLABLE by design: records written by eval runs, the CLI, or any
    -- non-HTTP caller carry none of these, and that absence is meaningful.
    --
    -- request_id is minted server-side (uuid4 hex); the inbound x-request-id header is NOT
    -- reused, because a caller-supplied string written straight into this table is an
    -- unauthenticated write into the record store. user_agent, by contrast, IS caller-
    -- controlled text and is stored knowingly — that is what a User-Agent is.
    request_id        text,
    user_agent        text,
    -- Salted SHA-256 of the client address (src/provenance/records.py:hash_client).
    -- A RAW IP MUST NEVER BE WRITTEN TO THIS COLUMN. 64 lowercase hex chars, enforced:
    client_hash       text check (client_hash ~ '^[0-9a-f]{64}$'),

    -- Server-side insert time. Distinct from "timestamp" (which the app sets) so a
    -- back-dated or wrong-clock record is detectable rather than authoritative.
    inserted_at       timestamptz      not null default now()
);

create index if not exists answer_records_timestamp_idx  on public.answer_records ("timestamp" desc);
create index if not exists answer_records_client_hash_idx on public.answer_records (client_hash);

-- Governance: the table is written by one server-side process and read by the verifier.
-- RLS on with NO policy means anon/authenticated keys can do nothing at all; the
-- service_role key bypasses RLS and is the only thing that works. Consequence, stated so it
-- is a decision and not an accident: PROVENANCE_RECORDS_KEY is a service_role secret. It
-- belongs in the API project's env only — never in a NEXT_PUBLIC_* var, never in web/.
alter table public.answer_records enable row level security;

-- Verify what landed (Day 2's `--verify` does the real consistency checks):
--   select run_id, "timestamp", question, confidence, request_id, client_hash
--   from public.answer_records order by "timestamp" desc limit 20;
