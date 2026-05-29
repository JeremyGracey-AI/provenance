"use client";

import { useState } from "react";
import { GroundedAnswer, PageRef, VerifiedClaim, pageId } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const EXAMPLES = [
  "What are the four primary tissue types?",
  "Describe the sliding filament model of muscle contraction.",
  "Which bones form the orbit of the eye?",
  "What are the phases of an action potential?",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GroundedAnswer | null>(null);

  async function ask(q: string) {
    const question = q.trim();
    if (!question || loading) return;
    setQuery(question);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });
      if (!res.ok) throw new Error(`API responded ${res.status}`);
      setResult((await res.json()) as GroundedAnswer);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="wrap">
      <header className="masthead">
        <h1 className="wordmark">
          Provenance<span className="dot">.</span>
        </h1>
        <p className="tagline">
          Ask the OpenStax <em>Anatomy &amp; Physiology</em> textbook anything. Every claim is
          checked against the exact page image that proves it — and shown to you.
        </p>
      </header>

      <form
        className="search"
        onSubmit={(e) => {
          e.preventDefault();
          ask(query);
        }}
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. What is the function of the loop of Henle?"
          aria-label="Ask a question"
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Reading…" : "Ask"}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => ask(ex)} type="button">
            {ex}
          </button>
        ))}
      </div>

      {loading && <p className="note">Retrieving pages, drafting an answer, and verifying every claim…</p>}
      {error && <p className="error">Couldn’t reach the API ({error}). Is the backend running?</p>}

      {result && <Result data={result} />}

      <footer className="foot">
        Corpus: OpenStax <em>Anatomy &amp; Physiology 2e</em> (CC&nbsp;BY 4.0), 1,347 pages.
        Retrieval by Cohere Embed&nbsp;v4 over page images; answers &amp; verification by Claude.
        Page numbers are PDF page indices. A builder-track demo — metrics are reported honestly.
      </footer>
    </main>
  );
}

function Result({ data }: { data: GroundedAnswer }) {
  const urlById = new Map<string, string>();
  for (const p of data.retrieved) {
    if (p.image_url) urlById.set(pageId(p), p.image_url);
  }
  const supported = data.claims.filter((c) => c.verdict === "supported").length;
  const pct = Math.round(data.confidence * 100);

  return (
    <section>
      <div className="answer-head">
        <p className="answer">{data.answer}</p>
        <div className="confidence" title={`${supported} of ${data.claims.length} claims grounded`}>
          <span className="pct">{pct}%</span>
          <span className="label">
            {supported}/{data.claims.length} grounded
          </span>
        </div>
      </div>

      <p className="section-label">
        Claims &amp; their evidence{data.repairs > 0 ? ` · ${data.repairs} repair pass` : ""}
      </p>
      {data.claims.map((claim, i) => (
        <ClaimRow key={i} claim={claim} urlById={urlById} />
      ))}

      <p className="section-label">Pages retrieved</p>
      <div className="sources">
        {data.retrieved.map((p) => (
          <span className="source-pill" key={pageId(p)}>
            <b>p{p.page_number}</b> · {p.score.toFixed(3)}
          </span>
        ))}
      </div>
    </section>
  );
}

function ClaimRow({ claim, urlById }: { claim: VerifiedClaim; urlById: Map<string, string> }) {
  const cited = claim.citations
    .map((id) => ({ id, url: urlById.get(id), n: id.split("#p")[1] }))
    .filter((c) => c.url);

  return (
    <div className="claim">
      <div>
        <span className={`verdict ${claim.verdict}`}>
          {claim.verdict === "supported" ? "✓ supported" : "✕ unsupported"}
        </span>
        <p className="claim-text">{claim.text}</p>
        {claim.evidence && <p className="evidence">“{claim.evidence}”</p>}
      </div>
      <div className="pages">
        {cited.map((c) => (
          <figure className="page-fig" key={c.id}>
            <a href={c.url} target="_blank" rel="noreferrer">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={c.url} alt={`Textbook page ${c.n}`} loading="lazy" />
            </a>
            <figcaption>page {c.n}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
