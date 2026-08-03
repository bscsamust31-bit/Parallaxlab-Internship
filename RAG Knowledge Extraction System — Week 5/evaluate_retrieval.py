"""
evaluate_retrieval.py — Precision@K / Recall@K evaluation for the retrieval stage.

Ground truth: since no human relevance judgments exist for this corpus, the test set uses
each ArXiv category as a cheap-but-principled proxy for "relevant documents" -- a query written
about a specific category's subject matter is expected to retrieve documents *from* that
category. This is the same signal Week 4 already validated independently (topic modeling
clustered cleanly along these same category lines), so treating "same category as the query's
topic" as ground-truth relevance is a reasonable, defensible proxy in the absence of manual
relevance labels.

Run with:
    python evaluate_retrieval.py
"""

import json
from pathlib import Path

import pandas as pd

from rag_pipeline import RAGPipeline

K_VALUES = [1, 3, 5, 10]

# (query, relevant_category) -- "relevant" = every doc_id in that category
TEST_SET = [
    ("transformer models and self-attention for natural language understanding", "cs.CL"),
    ("language model architecture for text classification", "cs.CL"),
    ("convolutional neural networks for image classification", "cs.CV"),
    ("deep residual networks for computer vision", "cs.CV"),
    ("stochastic gradient descent convergence for deep learning", "cs.LG"),
    ("optimization and learning rate schedules in neural networks", "cs.LG"),
    ("dense retrieval and semantic search with bi-encoders", "cs.IR"),
    ("contrastive learning for document retrieval", "cs.IR"),
    ("reinforcement learning for robotic manipulation", "cs.RO"),
    ("robot control policies trained via simulation", "cs.RO"),
]


def precision_recall_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    top_k = retrieved_doc_ids[:k]
    hits = len(set(top_k) & relevant_doc_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    return precision, recall


def main():
    pipeline = RAGPipeline()
    pipeline.load()

    if "category" not in pipeline.chunks_df.columns:
        raise RuntimeError("chunks_df has no 'category' column -- cannot build ground truth.")

    relevant_by_category = {
        cat: set(group["doc_id"].tolist())
        for cat, group in pipeline.chunks_df.groupby("category")
    }

    rows = []
    max_k = max(K_VALUES)
    for query, relevant_category in TEST_SET:
        relevant_ids = relevant_by_category.get(relevant_category, set())
        retrieved = pipeline.semantic_search(query, k=max_k)
        retrieved_doc_ids = retrieved["doc_id"].tolist() if not retrieved.empty else []

        row = {"query": query, "relevant_category": relevant_category, "n_relevant": len(relevant_ids)}
        for k in K_VALUES:
            p, r = precision_recall_at_k(retrieved_doc_ids, relevant_ids, k)
            row[f"precision@{k}"] = round(p, 3)
            row[f"recall@{k}"] = round(r, 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    results_df.to_csv(reports_dir / "retrieval_eval_report.csv", index=False)

    summary = {f"mean_precision@{k}": round(results_df[f"precision@{k}"].mean(), 3) for k in K_VALUES}
    summary.update({f"mean_recall@{k}": round(results_df[f"recall@{k}"].mean(), 3) for k in K_VALUES})
    with open(reports_dir / "retrieval_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated {len(TEST_SET)} queries against {len(relevant_by_category)} categories.\n")
    print(results_df.to_string(index=False))
    print("\n--- Summary (mean across all queries) ---")
    for k in K_VALUES:
        print(f"  K={k:>2}  Precision@K={summary[f'mean_precision@{k}']:.3f}  "
              f"Recall@K={summary[f'mean_recall@{k}']:.3f}")
    print(f"\nSaved: {reports_dir / 'retrieval_eval_report.csv'}, "
          f"{reports_dir / 'retrieval_eval_summary.json'}")


if __name__ == "__main__":
    main()
