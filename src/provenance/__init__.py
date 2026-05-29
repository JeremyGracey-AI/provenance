"""Provenance: citation-grounded RAG over textbook page images.

Retrieve the pages that answer a question, draft an answer, then verify every claim
against the exact page that proves it. Backends are swappable behind narrow protocols,
so the offline fakes (used in CI) and the real Cohere + Claude stack share one graph.
"""

__version__ = "0.1.0"
