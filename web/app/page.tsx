"use client";

import { useEffect, useMemo, useState } from "react";
import {
  GroundedAnswer,
  PageRef,
  VerifiedClaim,
  pageId,
} from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "What happens during depolarization in an action potential?",
  "Explain the sliding filament model of contraction.",
  "What is the function of the loop of Henle?",
];

const LOAD_STEPS = [
  "retrieving pages",
  "reading pages",
  "verifying claims",
  "checking citations",
];

type AppState = "entry" | "loading" | "result" | "error";

export default function Home() {
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GroundedAnswer | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);
  const [activeClaim, setActiveClaim] = useState(0);
  const [loadStep, setLoadStep] = useState(0);

  /* ----- Source pages to display in the viewer.
     Prefer pages that any claim actually cites, falling back to the full
     retrieved set if no claim has citations yet. Preserves first-seen order. */
  const viewerPages = useMemo<PageRef[]>(() => {
    if (!result) return [];
    const byId = new Map<string, PageRef>();
    for (const p of result.retrieved) byId.set(pageId(p), p);

    const seen = new Set<string>();
    const cited: PageRef[] = [];
    for (const c of result.claims) {
      for (const cid of c.citations) {
        if (seen.has(cid)) continue;
        const ref = byId.get(cid);
        if (!ref) continue;
        seen.add(cid);
        cited.push(ref);
      }
    }
    return cited.length ? cited : result.retrieved;
  }, [result]);

  /* ----- Cycle through the loading status labels. */
  useEffect(() => {
    if (!pending) return;
    setLoadStep(0);
    let s = 0;
    const iv = window.setInterval(() => {
      s = Math.min(s + 1, LOAD_STEPS.length - 1);
      setLoadStep(s);
    }, 480);
    return () => window.clearInterval(iv);
  }, [pending]);

  async function ask(q: string) {
    const question = q.trim();
    if (!question || pending) return;
    setQuery(question);
    setPending(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });
      if (!res.ok) throw new Error(`API responded ${res.status}`);
      const data = (await res.json()) as GroundedAnswer;
      setResult(data);
      setActiveIdx(0);
      setActiveClaim(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setPending(false);
    }
  }

  function focusPage(pid: string) {
    const i = viewerPages.findIndex((p) => pageId(p) === pid);
    if (i >= 0) setActiveIdx(i);
  }

  function selectClaim(i: number) {
    setActiveClaim(i);
    const c = result?.claims[i];
    if (c && c.citations.length) focusPage(c.citations[0]);
  }

  const state: AppState = pending
    ? "loading"
    : error
      ? "error"
      : result
        ? "result"
        : "entry";

  return (
    <>
      <header className="site">
        <div className="wrap">
          <div className="brand">
            <span className="wordmark">
              Provenance<span className="dot">.</span>
            </span>
            <span className="tag">Visual-Grounded Document QA</span>
          </div>
          <div className="corpus-chip" aria-label="Corpus">
            <BookIcon />
            <span className="hide-sm">CORPUS</span>{" "}
            <b>Anatomy &amp; Physiology 2e</b>
            <span className="hide-sm">&nbsp;· 1,347 pp</span>
          </div>
        </div>
      </header>

      {state !== "entry" && (
        <div className="qbar">
          <div className="wrap">
            <form
              className="ask-again"
              onSubmit={(e) => {
                e.preventDefault();
                ask(query);
              }}
            >
              <SearchIcon />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask another question…"
                aria-label="Ask another question"
              />
            </form>
            <button
              className="ask-btn"
              type="button"
              onClick={() => ask(query)}
              disabled={pending}
            >
              Ask <ArrowIcon />
            </button>
          </div>
        </div>
      )}

      <main>
        <div className="wrap">
          {state === "entry" && (
            <section className="console-stage">
              <div className="console">
                <div className="label overline reveal d1">Ask the textbook</div>
                <h1 className="reveal d2">
                  Answers you can <em>trace</em> to the page.
                </h1>
                <p className="sub reveal d2">
                  Provenance reads the textbook by sight, drafts an answer, then
                  verifies every claim against the exact page it came from.
                </p>

                <form
                  className="field reveal d3"
                  onSubmit={(e) => {
                    e.preventDefault();
                    ask(query);
                  }}
                >
                  <input
                    autoFocus
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask a question about human anatomy or physiology…"
                    aria-label="Ask a question"
                  />
                  <button className="ask-btn" type="submit" disabled={pending}>
                    Ask <ArrowIcon />
                  </button>
                </form>

                <div className="examples reveal d3">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex}
                      className="ex"
                      type="button"
                      onClick={() => ask(ex)}
                    >
                      {ex}
                    </button>
                  ))}
                </div>

                <p className="promise reveal d4">
                  Claims that <b>can&rsquo;t be grounded are flagged</b>, not hidden.
                  Every answer shows its sources.
                </p>
              </div>
            </section>
          )}

          {state === "loading" && (
            <section className="loading show">
              <div className="load-status">
                <span className="dot-pulse" />
                <span>{LOAD_STEPS[loadStep]}</span>
              </div>
              <div className="grid">
                <div>
                  <div className="skl" style={{ height: 150, marginBottom: 20 }} />
                  <div className="skl" style={{ height: 92, marginBottom: 12 }} />
                  <div className="skl" style={{ height: 92, marginBottom: 12 }} />
                </div>
                <div>
                  <div className="skl" style={{ aspectRatio: "612 / 792" }} />
                </div>
              </div>
            </section>
          )}

          {state === "error" && (
            <section className="result show">
              <div className="block">
                <div className="block-head">
                  <span className="label">
                    <span className="sec-num">!!</span> — Couldn&rsquo;t reach the API
                  </span>
                </div>
                <div className="answer-card">
                  <p className="answer-text">{error}</p>
                  <p className="answer-text" style={{ marginTop: 14, fontSize: 15, color: "var(--ink-soft)" }}>
                    The backend at <span className="mono">{API_URL}</span> didn&rsquo;t respond.
                    If you&rsquo;re running it locally, make sure it&rsquo;s up; if it&rsquo;s a
                    hosted deployment, check the URL configured at build time.
                  </p>
                </div>
              </div>
            </section>
          )}

          {state === "result" && result && (
            <section className="result show">
              <div className="grid">
                <div>
                  <AnswerBlock data={result} />
                  <ClaimsBlock
                    data={result}
                    activeClaim={activeClaim}
                    onSelectClaim={selectClaim}
                    onFocusPage={focusPage}
                  />
                </div>
                <aside>
                  <div className="viewer">
                    <div className="block-head">
                      <span className="label">
                        <span className="sec-num">03</span> — Source pages
                      </span>
                    </div>
                    <SourceViewer
                      pages={viewerPages}
                      activeIdx={activeIdx}
                      onSelect={setActiveIdx}
                    />
                  </div>
                </aside>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="site">
        <div className="wrap">
          <span className="fnote">
            Provenance · grounded document QA · a{" "}
            <a href="https://jeremygracey.ai">jeremygracey.ai</a> project
          </span>
          <span className="fnote">
            Corpus: OpenStax Anatomy &amp; Physiology 2e (CC&nbsp;BY)
          </span>
        </div>
      </footer>
    </>
  );
}

/* ----------------------------------------------------------------
   Answer card with grounded-confidence meter and retrieval meta.
---------------------------------------------------------------- */
function AnswerBlock({ data }: { data: GroundedAnswer }) {
  const pct = Math.round((data.confidence ?? 0) * 100);
  const supported = data.claims.filter((c) => c.verdict === "supported").length;

  return (
    <div className="block">
      <div className="block-head">
        <span className="label">
          <span className="sec-num">01</span> — Answer
        </span>
      </div>

      <div className="answer-card">
        <div className="meter">
          <div className="meter-top">
            <span className="label">Grounded confidence</span>
            <span className="meter-pct">{pct}%</span>
          </div>
          <div className="meter-track">
            <div className="meter-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="meter-note">
            {supported} of {data.claims.length}{" "}
            {data.claims.length === 1 ? "claim" : "claims"} verified against the source
          </div>
        </div>

        <p className="answer-text">{data.answer}</p>

        <div className="retrieval-meta">
          <span>
            <b>RETRIEVAL</b> Embed&nbsp;v4 · k={data.retrieved.length}
          </span>
          <span>
            <b>LOOP</b> {data.repairs} repair{data.repairs === 1 ? "" : "es"}
          </span>
          <span>
            <b>VERIFIER</b> Claude vision
          </span>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------
   Claims ledger — each claim is a clickable card that focuses the
   viewer on its first cited page.
---------------------------------------------------------------- */
function ClaimsBlock({
  data,
  activeClaim,
  onSelectClaim,
  onFocusPage,
}: {
  data: GroundedAnswer;
  activeClaim: number;
  onSelectClaim: (i: number) => void;
  onFocusPage: (pid: string) => void;
}) {
  const pageNumByCid = new Map<string, number>();
  for (const p of data.retrieved) pageNumByCid.set(pageId(p), p.page_number);

  return (
    <div className="block">
      <div className="block-head">
        <span className="label">
          <span className="sec-num">02</span> — Claims ledger
        </span>
      </div>
      <div>
        {data.claims.map((c, i) => (
          <ClaimCard
            key={i}
            claim={c}
            active={i === activeClaim}
            onClick={() => onSelectClaim(i)}
            onFocusPage={onFocusPage}
            pageNumByCid={pageNumByCid}
          />
        ))}
      </div>
    </div>
  );
}

function ClaimCard({
  claim,
  active,
  onClick,
  onFocusPage,
  pageNumByCid,
}: {
  claim: VerifiedClaim;
  active: boolean;
  onClick: () => void;
  onFocusPage: (pid: string) => void;
  pageNumByCid: Map<string, number>;
}) {
  const k = claim.verdict === "supported" ? "sup" : "uns";
  return (
    <div
      className={`claim ${k} ${active ? "active" : ""}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div className="claim-top">
        <span className={`pill ${k}`}>
          {claim.verdict === "supported" ? <CheckIcon /> : <WarnIcon />}
          {claim.verdict === "supported" ? "Supported" : "Unsupported"}
        </span>
      </div>
      <div className="claim-text">{claim.text}</div>
      {claim.evidence && (
        <div className="evidence">
          <span className="ekey">
            {claim.verdict === "supported" ? "Evidence" : "Why not"}
          </span>
          {claim.evidence}
        </div>
      )}
      {claim.citations.length > 0 && (
        <div className="pagechips">
          {claim.citations.map((cid) => {
            const num = pageNumByCid.get(cid);
            if (num === undefined) return null;
            return (
              <button
                key={cid}
                type="button"
                className="pchip"
                onClick={(e) => {
                  e.stopPropagation();
                  onFocusPage(cid);
                }}
              >
                p. {num}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------
   Source page viewer — sticky on desktop, large active page plus
   thumbnail strip for the other cited pages.
---------------------------------------------------------------- */
function SourceViewer({
  pages,
  activeIdx,
  onSelect,
}: {
  pages: PageRef[];
  activeIdx: number;
  onSelect: (i: number) => void;
}) {
  const active = pages[activeIdx];
  if (!active) return null;

  return (
    <div className="frame">
      <div className="page-mat">
        {active.image_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={active.image_url}
            src={active.image_url}
            alt={`Textbook page ${active.page_number}`}
            className="page-img"
            loading="eager"
          />
        ) : (
          <div className="page-placeholder">No image available for this page.</div>
        )}
      </div>
      <div className="pcap">
        <span className="pid">
          Page <b>{active.page_number}</b>
        </span>
        <span className="score">score {active.score.toFixed(3)}</span>
      </div>
      {pages.length > 1 && (
        <div className="thumbs">
          {pages.map((p, i) => (
            <button
              key={pageId(p)}
              type="button"
              className={`thumb ${i === activeIdx ? "on" : ""}`}
              onClick={() => onSelect(i)}
              title={`Page ${p.page_number}`}
              aria-label={`Show page ${p.page_number}`}
            >
              {p.image_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={p.image_url} alt="" loading="lazy" className="thumb-img" />
              ) : null}
              <span className="tnum">p. {p.page_number}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------
   Inline SVG icons (no extra dependency).
---------------------------------------------------------------- */
function BookIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      aria-hidden
    >
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={3}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function WarnIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
    </svg>
  );
}
