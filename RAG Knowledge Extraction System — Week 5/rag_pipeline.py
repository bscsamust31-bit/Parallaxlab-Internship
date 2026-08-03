"""
rag_pipeline.py — Core retrieval + generation logic for the RAG Knowledge Extraction System.

This module is intentionally framework-agnostic: it has no FastAPI, pytest, or CLI code in it.
main.py (the API), evaluate_retrieval.py, evaluate_generation.py, and test_main.py all import
from here, so the retrieval/generation logic is defined exactly once and tested exactly once.

Carries forward Week 2 (chunking + embeddings + ChromaDB + semantic_search), Week 3
(prompting + DeepSeek/OpenRouter generation + error handling + hallucination checks), and
Week 4 (topic_id / entities metadata, used here for optional filtered retrieval).
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from the current working directory, if one exists
except ImportError:
    pass  # python-dotenv not installed -- .env just won't be picked up automatically;
          # OPENROUTER_API_KEY set directly in the shell still works fine.

logger = logging.getLogger("rag_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
CHROMA_PATH = os.environ.get("RAG_CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "arxiv_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL_NAME = "deepseek/deepseek-v4-flash"

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30
MAX_CONTEXT_TOKENS = 1200
CHARS_PER_TOKEN_ESTIMATE = 4
MIN_GROUNDEDNESS_OVERLAP = 0.15
MIN_RELEVANCE_SCORE = 0.12
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")

SYSTEM_PROMPT = (
    "You are a research assistant answering questions using ONLY the numbered sources "
    "provided below. Follow these rules strictly:\n"
    "1. Base every claim in your answer on the provided sources -- do not use outside knowledge.\n"
    "2. Cite the source(s) you used inline, like [Source 2], for every factual claim.\n"
    "3. If the sources do not contain enough information to answer, say so explicitly instead "
    "of guessing.\n"
    "4. Keep answers concise (3-6 sentences) and directly address the question."
)

OUT_OF_DOMAIN_RESPONSE = (
    "I don't have enough relevant information in this knowledge base to answer that question "
    "confidently. This corpus covers ArXiv computer science / ML research papers -- try "
    "rephrasing your question around one of those topics."
)


class RAGNotReadyError(RuntimeError):
    """Raised when the corpus/collection can't be loaded -- lets callers (e.g. the API's
    error handler) turn this into a proper 500 with a clear message instead of an opaque crash."""


def _build_fallback_corpus() -> pd.DataFrame:
    """Small, schema-matching synthetic corpus, identical in spirit to Weeks 2-4's fallback,
    so this module (and everything built on it) is runnable and testable without Weeks 1-2
    having been executed first."""
    topics_seed = {
        "cs.CL": ("Transformer Language Model", "This paper introduces a transformer-based "
                  "architecture for natural language understanding. Self-attention mechanisms "
                  "allow the model to capture long-range dependencies between tokens far more "
                  "effectively than recurrent networks."),
        "cs.CV": ("Convolutional Image Classification", "We propose a convolutional neural "
                  "network architecture for large-scale image classification using residual "
                  "connections to enable training of very deep architectures."),
        "cs.LG": ("Gradient Based Optimization", "This work analyzes the convergence properties "
                  "of stochastic gradient descent under non-convex loss landscapes typical of "
                  "deep neural networks."),
        "cs.IR": ("Dense Retrieval Semantic Search", "We present a dense retrieval system that "
                  "encodes queries and documents into a shared embedding space using a "
                  "bi-encoder trained with contrastive loss."),
        "cs.RO": ("Robot Manipulation Reinforcement Learning", "This paper studies reinforcement "
                  "learning for robotic manipulation tasks in cluttered environments, training a "
                  "policy that generalizes to novel object configurations."),
    }
    rows = []
    for cat, (title_seed, para) in topics_seed.items():
        for i in range(15):
            text = (para + " ") * 2
            rows.append({
                "chunk_id": f"synthetic.{cat}.{i:04d}::chunk0",
                "doc_id": f"synthetic.{cat}.{i:04d}",
                "chunk_index": 0, "text": text.strip(), "char_count": len(text),
                "title": f"{title_seed} — Study {i + 1}", "category": cat,
                "published": f"2024-0{(i % 9) + 1}-15T00:00:00Z",
            })
    return pd.DataFrame(rows)


class RAGPipeline:
    """Holds the loaded corpus, ChromaDB collection, and embedding backend, and exposes
    retrieval + generation as plain methods. Instantiate once (e.g. at API startup) and reuse."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL_NAME):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model_name = model_name
        self.chunks_df: Optional[pd.DataFrame] = None
        self.collection = None
        self.embedding_backend: Optional[str] = None
        self._embed_fn = None
        self._loaded = False

    # -- Setup -----------------------------------------------------------
    def load(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        enriched_path = DATA_DIR / "chunks_enriched.parquet"
        plain_path = DATA_DIR / "chunks.parquet"

        try:
            if enriched_path.exists():
                self.chunks_df = pd.read_parquet(enriched_path)
            elif plain_path.exists():
                self.chunks_df = pd.read_parquet(plain_path)
            else:
                logger.warning("No chunked corpus found at %s -- using offline fallback corpus.",
                                DATA_DIR)
                self.chunks_df = _build_fallback_corpus()
                self.chunks_df.to_parquet(plain_path, index=False)
        except Exception as e:
            raise RAGNotReadyError(f"Failed to load corpus: {e}") from e

        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise RAGNotReadyError(f"Failed to connect to ChromaDB at {CHROMA_PATH}: {e}") from e

        self._setup_embedding_backend()

        if self.collection.count() != len(self.chunks_df):
            self._ingest(self.chunks_df)

        self._loaded = True
        logger.info("RAGPipeline ready: %d chunks, %d vectors, backend=%s",
                    len(self.chunks_df), self.collection.count(), self.embedding_backend)

    def _setup_embedding_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.embedding_backend = f"sentence-transformers ({EMBEDDING_MODEL_NAME})"

            def embed(texts):
                return np.asarray(model.encode(list(texts), normalize_embeddings=True))
            _ = embed(["smoke test"])
            self._embed_fn = embed
        except Exception as e:
            logger.warning("Could not load '%s' (%s) -- falling back to TF-IDF + SVD.",
                            EMBEDDING_MODEL_NAME, type(e).__name__)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize as sk_normalize

            vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
            tfidf = vectorizer.fit_transform(self.chunks_df["text"].tolist())
            n_components = min(384, tfidf.shape[1] - 1, self.chunks_df.shape[0] - 1)
            svd = TruncatedSVD(n_components=max(n_components, 2), random_state=42)
            svd.fit(tfidf)
            self.embedding_backend = "TF-IDF + TruncatedSVD (offline fallback)"

            def embed(texts):
                return sk_normalize(svd.transform(vectorizer.transform(list(texts))))
            self._embed_fn = embed

    def _ingest(self, chunks_df: pd.DataFrame) -> None:
        embeddings = self._embed_fn(chunks_df["text"].tolist())
        metas = [
            {"doc_id": r.doc_id, "chunk_index": int(r.chunk_index), "title": r.title,
             "category": r.category, "published": str(r.published)}
            for r in chunks_df.itertuples(index=False)
        ]
        for start in range(0, len(chunks_df), 500):
            end = start + 500
            self.collection.upsert(
                ids=chunks_df["chunk_id"].tolist()[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=chunks_df["text"].tolist()[start:end],
                metadatas=metas[start:end],
            )

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.collection is not None

    # -- Retrieval (Week 2) ------------------------------------------------
    def semantic_search(self, query: str, k: int = 5, category: Optional[str] = None,
                         topic_id: Optional[int] = None) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["rank", "score", "chunk_id", "doc_id", "title", "category", "text"])
        if not query or not query.strip() or self.collection.count() == 0:
            return empty

        conditions = []
        if category is not None:
            conditions.append({"category": category})
        if topic_id is not None:
            conditions.append({"topic_id": topic_id})
        where = {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else None)

        query_embedding = np.asarray(self._embed_fn([query]))[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(k, self.collection.count()),
            where=where, include=["documents", "metadatas", "distances"],
        )

        rows = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(
                results.get("ids", [[]])[0], results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]), start=1):
            rows.append({
                "rank": rank, "score": round(1 - dist, 4), "chunk_id": cid,
                "doc_id": meta.get("doc_id", ""), "title": meta.get("title", ""),
                "category": meta.get("category", ""), "text": doc,
            })
        return pd.DataFrame(rows)

    # -- Prompting (Week 3) ------------------------------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    def _truncate_context(self, retrieved: pd.DataFrame, max_tokens: int = MAX_CONTEXT_TOKENS) -> pd.DataFrame:
        if retrieved.empty:
            return retrieved
        kept, running = [], 0
        for _, row in retrieved.iterrows():
            t = self._estimate_tokens(row["text"])
            if running + t > max_tokens and kept:
                break
            kept.append(row)
            running += t
        return pd.DataFrame(kept).reset_index(drop=True) if kept else retrieved.iloc[:1]

    @staticmethod
    def _format_context(retrieved: pd.DataFrame) -> str:
        if retrieved.empty:
            return "(no sources retrieved)"
        blocks = []
        for _, row in retrieved.iterrows():
            blocks.append(f"[Source {row['rank']}] (doc_id={row['doc_id']}, title=\"{row['title']}\")\n{row['text']}")
        return "\n\n".join(blocks)

    def _build_messages(self, query: str, retrieved: pd.DataFrame) -> List[dict]:
        user_content = (
            f"SOURCES:\n{self._format_context(retrieved)}\n\nQUESTION: {query}\n\n"
            "Answer the question using only the sources above, with inline [Source N] citations."
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    @staticmethod
    def _offline_fallback_answer(retrieved: pd.DataFrame) -> str:
        if retrieved.empty:
            return "[offline fallback -- no LLM call made] I don't have enough retrieved context to answer this question."
        lines = ["[offline fallback -- no LLM call made] Based on the retrieved sources:"]
        for _, row in retrieved.head(3).iterrows():
            snippet = row["text"][:220] + ("..." if len(row["text"]) > 220 else "")
            lines.append(f"- {snippet} [Source {row['rank']}]")
        return "\n".join(lines)

    # -- Generation (Week 3) -------------------------------------------------
    def _call_llm(self, messages: List[dict], retrieved: pd.DataFrame) -> dict:
        if not self.api_key:
            return {"answer": self._offline_fallback_answer(retrieved),
                    "backend": "offline fallback (no API key)", "retries": 0, "error": None}

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model_name, "messages": messages, "temperature": 0.2},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2 ** attempt)
                    last_error = f"429 rate limited (attempt {attempt + 1})"
                    time.sleep(delay + random.uniform(0, 0.5))
                    continue
                if resp.status_code >= 500:
                    last_error = f"{resp.status_code} server error (attempt {attempt + 1})"
                    time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))
                    continue
                if resp.status_code >= 400:
                    last_error = f"{resp.status_code} client error: {resp.text[:300]}"
                    break
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                return {"answer": answer, "backend": f"{self.model_name} (OpenRouter)",
                        "retries": attempt, "error": None}
            except requests.exceptions.Timeout:
                last_error = f"timeout after {REQUEST_TIMEOUT_SECONDS}s (attempt {attempt + 1})"
                time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))
            except requests.exceptions.ConnectionError as e:
                last_error = f"connection error (attempt {attempt + 1}): {e}"
                time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_error = f"malformed API response: {e}"
                break

        return {"answer": self._offline_fallback_answer(retrieved),
                "backend": f"offline fallback (after error: {last_error})",
                "retries": MAX_RETRIES, "error": last_error}

    @staticmethod
    def _groundedness_overlap(answer: str, retrieved: pd.DataFrame) -> float:
        answer_words = set(w.lower() for w in _WORD_RE.findall(answer))
        if not answer_words:
            return 1.0
        context_words = set(w.lower() for w in _WORD_RE.findall(" ".join(retrieved["text"].tolist())))
        if not context_words:
            return 0.0
        return len(answer_words & context_words) / len(answer_words)

    # -- End-to-end RAG (Week 3) ---------------------------------------------
    def answer(self, query: str, k: int = 5, category: Optional[str] = None,
               topic_id: Optional[int] = None, min_relevance: float = MIN_RELEVANCE_SCORE) -> dict:
        t0 = time.perf_counter()
        retrieved = self.semantic_search(query, k=k, category=category, topic_id=topic_id)
        t_retrieved = time.perf_counter()

        if retrieved.empty or float(retrieved["score"].max()) < min_relevance:
            return {
                "query": query, "answer": OUT_OF_DOMAIN_RESPONSE, "backend": "relevance-gate (no LLM call)",
                "sources": [], "out_of_domain": True, "possible_hallucination": False,
                "groundedness_overlap": None, "retries": 0,
                "retrieval_ms": round((t_retrieved - t0) * 1000, 2), "generation_ms": 0.0,
                "total_ms": round((t_retrieved - t0) * 1000, 2),
            }

        context = self._truncate_context(retrieved)
        messages = self._build_messages(query, context)
        llm_result = self._call_llm(messages, context)
        t_generated = time.perf_counter()
        overlap = self._groundedness_overlap(llm_result["answer"], context)

        return {
            "query": query, "answer": llm_result["answer"], "backend": llm_result["backend"],
            "sources": context[["rank", "score", "doc_id", "title", "category"]].to_dict("records"),
            "out_of_domain": False, "possible_hallucination": overlap < MIN_GROUNDEDNESS_OVERLAP,
            "groundedness_overlap": round(overlap, 3), "retries": llm_result["retries"],
            "retrieval_ms": round((t_retrieved - t0) * 1000, 2),
            "generation_ms": round((t_generated - t_retrieved) * 1000, 2),
            "total_ms": round((t_generated - t0) * 1000, 2),
        }