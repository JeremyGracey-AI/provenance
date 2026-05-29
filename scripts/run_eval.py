"""Evaluate the pipeline over the golden set.

    python scripts/run_eval.py                 # offline fakes; applies the plumbing gate (CI)
    python scripts/run_eval.py --real          # real Cohere + Claude; prints honest numbers

The default (fakes) is what CI runs: it proves routing, retrieval, answering, and the
verify→repair loop are wired correctly, then enforces PLUMBING_GATE (full faithfulness,
well-formed citations) and exits nonzero on failure. `--real` reports the headline metrics
(faithfulness, citation F1, recall@k, nDCG@k) and never gates — those numbers go in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from provenance.config import load_settings
from provenance.demo import demo_pipeline
from provenance.deploy import real_pipeline
from provenance.eval.dataset import load_golden
from provenance.eval.gate import PLUMBING_GATE, check_gate
from provenance.eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Provenance eval.")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/anatomy_golden.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--real", action="store_true", help="use the real Cohere + Claude pipeline")
    parser.add_argument("--k", type=int, default=None, help="override retrieval k (defaults to settings.top_k)")
    args = parser.parse_args()

    settings = load_settings()
    k = args.k or settings.top_k
    golden = load_golden(args.golden)
    pipeline = real_pipeline(settings, args.corpus) if args.real else demo_pipeline(settings)

    report = run_eval(pipeline, golden, k=k)
    print(report.table())

    if args.real:
        return  # headline numbers are informational, not a gate

    passed, failures = check_gate(report, PLUMBING_GATE)
    if not passed:
        print("\nPLUMBING GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nplumbing gate passed")


if __name__ == "__main__":
    main()
