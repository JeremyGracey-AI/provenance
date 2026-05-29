// Mirrors provenance.models.GroundedAnswer (the API's /query response).

export type Verdict = "supported" | "unsupported";

export interface PageRef {
  doc_id: string;
  page_number: number;
  score: number;
  image_path: string | null;
  image_url: string | null;
}

export interface VerifiedClaim {
  text: string;
  citations: string[]; // PageRef.id values, e.g. "anatomy-physiology-2e#p146"
  verdict: Verdict;
  evidence: string;
}

export interface GroundedAnswer {
  question: string;
  answer: string;
  claims: VerifiedClaim[];
  retrieved: PageRef[];
  confidence: number; // 0..1, fraction of claims the judge upheld
  repairs: number;
}

export const pageId = (p: PageRef): string => `${p.doc_id}#p${p.page_number}`;
