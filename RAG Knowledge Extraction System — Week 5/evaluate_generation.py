"""
evaluate_generation.py — Automated end-to-end evaluation of generation quality and latency.

Runs a batch of test queries -- mixing in-domain questions (should retrieve strong context and
get a grounded, cited answer) with deliberately out-of-domain ones (should trip the relevance
gate) -- through the full RAG pipeline (rag_pipeline.RAGPipeline.answer), and reports:

  - Latency: retrieval_ms / generation_ms / total_ms, mean + p95, per run and in aggregate.
  - Relevance-gate accuracy: did out-of-domain queries actually get gated, and in-domain ones not.
  - Groundedness: mean overlap score, and the hallucination-flag rate.
  - Citation coverage: fraction of generated answers that actually include a [Source N] citation,
    as required by the Week 3 prompt -- a cheap, concrete generation-quality signal that doesn't
    require a reference answer to compute.

Run with:
    python evaluate_generation.py
"""

import json
import re
from pathlib import Path

import pandas as pd

from rag_pipeline import RAGPipeline

N_REPEATS = 3
CITATION_PATTERN = re.compile(r"\[Source \d+\]")

TEST_QUERIES = [
    ("transformer models for natural language understanding", False),
    ("convolutional neural networks for image classification", False),
    ("convergence of stochastic gradient descent optimization", False),
    ("dense embeddings for semantic document retrieval", False),
    ("reinforcement learning for robot manipulation", False),
    ("what's a good recipe for pizza dough", True),
    ("who won the football match last night", True),
    ("best budget smartphones to buy in 2026", True),
]


def main():
    pipeline = RAGPipeline()
    pipeline.load()

    rows = []
    for query, expected_ood in TEST_QUERIES:
        for rep in range(N_REPEATS):
            result = pipeline.answer(query)
            has_citation = bool(CITATION_PATTERN.search(result["answer"]))
            rows.append({
                "query": query, "repeat": rep, "expected_out_of_domain": expected_ood,
                "out_of_domain": result["out_of_domain"],
                "gate_correct": result["out_of_domain"] == expected_ood,
                "possible_hallucination": result["possible_hallucination"],
                "groundedness_overlap": result["groundedness_overlap"],
                "has_citation": has_citation if not result["out_of_domain"] else None,
                "retries": result["retries"], "backend": result["backend"],
                "retrieval_ms": result["retrieval_ms"], "generation_ms": result["generation_ms"],
                "total_ms": result["total_ms"],
            })

    results_df = pd.DataFrame(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    results_df.to_csv(reports_dir / "generation_eval_raw.csv", index=False)

    per_query_summary = results_df.groupby("query").agg(
        expected_out_of_domain=("expected_out_of_domain", "first"),
        gate_accuracy=("gate_correct", "mean"),
        mean_retrieval_ms=("retrieval_ms", "mean"),
        mean_generation_ms=("generation_ms", "mean"),
        mean_total_ms=("total_ms", "mean"),
        hallucination_rate=("possible_hallucination", "mean"),
    ).round(3).reset_index()
    per_query_summary.to_csv(reports_dir / "generation_eval_summary.csv", index=False)

    in_domain = results_df[~results_df["expected_out_of_domain"]]
    overall = {
        "n_runs": len(results_df),
        "gate_accuracy": round(results_df["gate_correct"].mean(), 3),
        "mean_total_ms": round(results_df["total_ms"].mean(), 2),
        "p95_total_ms": round(results_df["total_ms"].quantile(0.95), 2),
        "mean_retrieval_ms": round(results_df["retrieval_ms"].mean(), 2),
        "mean_generation_ms": round(results_df["generation_ms"].mean(), 2),
        "in_domain_mean_groundedness_overlap": round(in_domain["groundedness_overlap"].mean(), 3),
        "in_domain_hallucination_rate": round(in_domain["possible_hallucination"].mean(), 3),
        "in_domain_citation_coverage": round(in_domain["has_citation"].mean(), 3),
    }
    with open(reports_dir / "generation_eval_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    print(f"Ran {len(TEST_QUERIES)} queries x {N_REPEATS} repeats = {len(results_df)} total runs.\n")
    print(per_query_summary.to_string(index=False))
    print("\n--- Overall summary ---")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {reports_dir / 'generation_eval_raw.csv'}, "
          f"{reports_dir / 'generation_eval_summary.csv'}, "
          f"{reports_dir / 'generation_eval_overall.json'}")


if __name__ == "__main__":
    main()
