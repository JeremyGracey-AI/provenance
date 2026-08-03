-- Provenance decision records — proposed durable table (Supabase / PostgREST).
--
-- THIS DDL HAS NEVER BEEN EXECUTED AGAINST ANY POSTGRESQL, LOCAL OR REMOTE, BY ANYONE.
-- Not by its author, and not by the 2026-08-03 invigilation, which could not run it either:
-- there is no psql, no postgres and no docker on this machine, so its "not examined" section
-- records the entire file as unverified. Every statement below — the types, the check
-- constraints, the regex classes, the index definitions — is REASONED, not tested. The first
-- person to paste it into a SQL editor is the first person to find out whether it parses.
-- Treat a syntax error as expected, not as a surprise.
--
-- STATUS: PROPOSAL. Creating the table, minting the key, and setting the Vercel env vars are
-- the human's call (WEEK-4.md Workstream A Day 2); this file exists so that call is one
-- paste, not a design session.
--
-- 2026-08-03, `[human]` ruling 7: this file now describes record_version 2 — a required
-- `answer` column and an `evidence` key inside `claims`. That changes nothing about the
-- paragraph above. Still nobody has run it, and the new column is as untested as the old ones.
-- One consequence of `if not exists` worth knowing before pasting: if a table by this name
-- ALREADY existed anywhere, this statement would silently do nothing rather than add `answer`.
-- Adding the column to a live table is an `alter table ... add column answer text` plus a
-- decision about what to put in the existing rows, and that is a human's call, not a paste.
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
    -- in `--verify`: two rows can never claim the same run.
    --
    -- What this does NOT buy, corrected 2026-08-03 (invigilation defect 15): it does not make
    -- a retried POST idempotent. That sentence described behaviour the writer cannot exhibit.
    -- `records.py:HttpSink.write` issues ONE `client.post` per record and no retry, and it
    -- sends `Prefer: return=minimal` with no `resolution=merge-duplicates`, so a re-POST of an
    -- existing run_id would come back 409 and be swallowed by the fail-open catch as a dropped
    -- record. Idempotency is a property of a writer that retries; this writer drops. Anything
    -- that DOES retry against this table (a backfill script, a queue) must send
    -- `Prefer: resolution=merge-duplicates` and own that choice itself.
    --
    -- Format: `--verify` requires 32 lowercase hex digits (records.py:_RUN_ID_RE). This column
    -- is plain `text` and does not enforce that. No CHECK is added, deliberately: it would be
    -- one more untested constraint in a file that has never been parsed by a database, and a
    -- constraint that fails to parse breaks the whole paste.
    run_id            text primary key,

    -- This table is shaped for record_version 2 (`[human]` ruling 7, 2026-08-03), which is
    -- what `build_record` writes. Named because it has a consequence: a v1 record — anything
    -- written before 2026-08-03, including `records/answers/answers-2026-08-02.jsonl` in this
    -- repository — has no `answer` and CANNOT be inserted here; the not-null below would
    -- reject it. `--verify` still reads v1 (it dispatches on the version); this table does
    -- not. Backfilling one means deciding what `answer` should say for a run whose answer was
    -- never recorded, which is a human's call and not a default.
    record_version    integer          not null,
    -- Quoted because `timestamp` is a type name in Postgres. It must keep this name: it is
    -- the JSON key, and JsonlSink's day-file naming derives from it.
    "timestamp"       timestamptz      not null,
    question          text             not null,
    -- The prose SERVED to the caller (`GroundedAnswer.answer`), new and REQUIRED at v2. Without
    -- it a row could satisfy every check in this file and every check in `--verify` while the
    -- caller had been told something the claims never said (2026-08-03 invigilation, defect 3).
    --
    -- READ THIS BEFORE GRANTING ANYONE SELECT ON THIS TABLE. It is the most disclosive column
    -- here and it changes the table's privacy profile, which is an accepted cost of the ruling
    -- and not an oversight: `question` is caller-authored text, but `answer` is MODEL OUTPUT
    -- OVER CALLER-CHOSEN QUESTIONS — longer, more specific, and capable of restating whatever
    -- a caller put in the question plus whatever the corpus says about it. A medical question
    -- and its answer, side by side, keyed by `client_hash`, is a materially different data set
    -- from a list of questions. The key that reaches it is service_role (see the RLS note at
    -- the bottom), so "who can read this" is exactly "who has that secret".
    --
    -- Rows also get bigger: this is the largest field in a record, and unbounded — `--verify`
    -- caps nothing here, on purpose (a cap in the verifier that the writer does not share is
    -- the fork `records.py:is_present` exists to prevent).
    --
    -- NOT NULL but NOT non-blank: `HostedVLM.answer` uses `payload.get("answer", "")`
    -- (src/provenance/backends/vlm.py:170), so an empty answer is a real 200, and a CHECK
    -- rejecting it here would refuse a row the API can legitimately produce.
    answer            text             not null,
    model             text             not null,
    k                 integer          not null check (k >= 1),
    retrieved         jsonb            not null,   -- [{id, score}] in retrieval order
    -- [{text, citations, verdict, verify_fallback, evidence}] — `evidence` is the span the
    -- judge quoted off the page (v2; "" for an unsupported claim, which is a value and not an
    -- absence). Before v2 the record carried the verdict without the reason for it.
    claims            jsonb            not null,
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
    -- Present-but-BLANK is not a third state. `--verify` rejects it ("present but empty —
    -- omit the field instead") and `build_record` no longer emits it, because both now share
    -- `records.py:is_present`. This constraint puts the same rule in the STORE so the row
    -- cannot be inserted here and fail verification later, when it is a human's DELETE
    -- instead of a dropped field — `verify_paths` verifies a SET, so one blank row fails the
    -- whole table. (This is the 2026-08-02 Day-1 gate's blocking defect: `User-Agent: " "`
    -- → HTTP 200 → `--verify` exit 1, and this column had no constraint at all. Over a real
    -- uvicorn/h11 stack the trigger byte is not " " — h11 strips OWS, so " " arrives as ""
    -- — it is `User-Agent: \xa0`, legal obs-text per RFC 9110, which h11 passes through
    -- untouched and Python's str.strip() calls blank. Verified on loopback, both bytes.)
    --
    -- `[^[:space:]]` and deliberately NOT `btrim(user_agent) <> ''`: btrim removes "a space
    -- by default" (PG docs, functions-string.html), so btrim(E'\t') is E'\t' and the naive
    -- form would ACCEPT a tab-only value that Python's str.strip() calls blank — the same
    -- two-definitions-of-present defect, rebuilt across the store/verifier boundary. This is
    -- the literal expansion of `\S` (functions-matching.html) and carries no backslash, so
    -- standard_conforming_strings cannot change what it means: at least one non-space char.
    --
    -- Residual gap, named because it is real and not hypothetical: [[:space:]] is
    -- locale/ctype-defined and in most UTF-8 locales U+00A0 is NOT a space, while Python's
    -- str.strip() removes it — the very byte above. So this check is a COARSER filter than
    -- `--verify`, not an equal one. Provenance itself cannot post such a row (is_present
    -- drops it before the sink); this constraint is the second line, for any other writer.
    -- A human who wants exact parity with str.strip() can substitute the class below, which
    -- adds every code point Python treats as whitespace that [[:space:]] may not — left
    -- commented because NOTHING HERE HAS BEEN RUN AGAINST A DATABASE and a check constraint
    -- that fails to parse breaks the whole paste:
    --   check (user_agent is null or user_agent ~
    --     '[^[:space:]\u001c-\u001f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]')
    user_agent        text check (user_agent is null or user_agent ~ '[^[:space:]]'),
    -- Salted SHA-256 of the client address (src/provenance/records.py:hash_client).
    -- A RAW IP MUST NEVER BE WRITTEN TO THIS COLUMN. 64 lowercase hex chars, enforced:
    --
    -- READ THIS BEFORE ANALYSING THE COLUMN: a NULL client_hash does NOT mean "no address was
    -- available". Address capture is caller-suppressible — `x-forwarded-for: " "` or `","`
    -- makes `_client_address` return None (src/provenance/api/app.py:62-65), `hash_client`
    -- then returns None, and the field is dropped from an otherwise valid record. So this
    -- column counts callers who did not suppress it, and any "% of answers identified" metric
    -- computed from it is a lower bound, not a rate.
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
--
-- That sentence carries more weight since v2 added `answer`. This table now holds, per row,
-- what an anonymous caller asked AND what the system told them, grouped by a stable
-- per-caller digest. Anyone holding the service_role key can read all of it; nothing here
-- redacts, samples, or expires. Retention is unset by design — naming it is a policy decision
-- this file is not authorised to take, so it is stated as a gap rather than defaulted.
alter table public.answer_records enable row level security;

-- Verify what landed (Day 2's `--verify` does the real consistency checks):
--   select run_id, "timestamp", question, confidence, request_id, client_hash
--   from public.answer_records order by "timestamp" desc limit 20;
-- `answer` is deliberately NOT in that select list: it is the disclosive column, and a
-- did-the-insert-work check does not need it. Read it when you mean to.
