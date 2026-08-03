"""
test_main.py — pytest unit tests for the FastAPI RAG service.

Run with:
    pytest test_main.py -v

Covers: a healthy /health check, valid /search and /query requests, the explicit 400/422
validation boundaries this week's brief asks for, and a simulated 500 to prove the global
exception handler never leaks internals to the client.
"""

import pytest
from fastapi.testclient import TestClient

import main
from rag_pipeline import RAGPipeline


@pytest.fixture(scope="module", autouse=True)
def loaded_pipeline():
    """Load the pipeline once for the whole test module (mirrors the FastAPI startup event,
    but run eagerly so tests don't depend on TestClient's startup-event timing)."""
    main.pipeline.load()
    yield main.pipeline


@pytest.fixture
def client():
    return TestClient(main.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0
    assert body["embedding_backend"] is not None


# ---------------------------------------------------------------------------
# /search — valid requests
# ---------------------------------------------------------------------------
def test_search_valid_query_returns_results(client):
    resp = client.post("/search", json={"query": "transformer attention mechanism", "k": 3})
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    assert len(results) <= 3
    if results:
        assert set(results[0].keys()) == {"rank", "score", "doc_id", "title", "category", "text"}


def test_search_respects_k(client):
    resp = client.post("/search", json={"query": "neural network optimization", "k": 2})
    assert resp.status_code == 200
    assert len(resp.json()) <= 2


def test_search_category_filter(client):
    resp = client.post("/search", json={"query": "research paper", "k": 5, "category": "cs.CL"})
    assert resp.status_code == 200
    for r in resp.json():
        assert r["category"] == "cs.CL"


# ---------------------------------------------------------------------------
# /query — valid requests
# ---------------------------------------------------------------------------
def test_query_in_domain_returns_grounded_answer(client):
    resp = client.post("/query", json={"query": "How do transformers use attention?", "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["out_of_domain"] is False
    assert body["answer"]
    assert body["total_ms"] >= 0
    assert body["retrieval_ms"] >= 0


def test_query_out_of_domain_triggers_relevance_gate(client):
    resp = client.post("/query", json={"query": "what's a good pizza dough recipe", "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["out_of_domain"] is True
    assert body["sources"] == []
    assert body["generation_ms"] == 0.0


def test_query_default_k_is_used_when_omitted(client):
    resp = client.post("/query", json={"query": "gradient descent optimization"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 400 — Bad Request (business-logic validation: well-formed request, invalid semantics)
# ---------------------------------------------------------------------------
def test_search_unknown_category_returns_400(client):
    resp = client.post("/search", json={"query": "transformers", "k": 3, "category": "not.a.real.category"})
    assert resp.status_code == 400
    assert "Unknown category" in resp.json()["detail"]


def test_query_unknown_category_returns_400(client):
    resp = client.post("/query", json={"query": "transformers", "category": "not.a.real.category"})
    assert resp.status_code == 400


def test_health_unavailable_returns_503_when_not_ready(client, monkeypatch):
    # Simulate the pipeline not being ready (e.g. startup failed) -- should be a clear 503,
    # not a 200 with misleading data or an unhandled crash.
    monkeypatch.setattr(main.pipeline, "_loaded", False)
    resp = client.get("/health")
    assert resp.status_code == 503
    monkeypatch.setattr(main.pipeline, "_loaded", True)


# ---------------------------------------------------------------------------
# 422 — Unprocessable Entity (schema / type validation)
# ---------------------------------------------------------------------------
def test_query_missing_field_returns_422(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_query_empty_string_returns_422(client):
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_whitespace_only_returns_422(client):
    # Caught by the custom field_validator, which raises inside Pydantic validation --
    # FastAPI surfaces validator errors as 422.
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 422


def test_query_wrong_type_returns_422(client):
    resp = client.post("/query", json={"query": "valid question", "k": "not-a-number"})
    assert resp.status_code == 422


def test_query_k_out_of_allowed_range_returns_422(client):
    resp = client.post("/query", json={"query": "valid question", "k": 999})
    assert resp.status_code == 422


def test_query_k_below_minimum_returns_422(client):
    resp = client.post("/query", json={"query": "valid question", "k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 500 — Internal Server Error (unexpected exception on an otherwise valid request ->
# generic response via the global handler, no internals leaked)
# ---------------------------------------------------------------------------
def test_query_internal_error_returns_500_without_leaking_details(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated database connection failure with secret internal path /etc/x")

    monkeypatch.setattr(main.pipeline, "answer", _boom)

    # TestClient's default raise_server_exceptions=True deliberately re-raises the original
    # exception in the test process even after our handler sends a real response -- that's
    # intentional Starlette behavior so bugs aren't silently hidden during testing. To inspect
    # what an actual HTTP client would receive (a clean 500 JSON body), disable that here.
    non_raising_client = TestClient(main.app, raise_server_exceptions=False)
    resp = non_raising_client.post("/query", json={"query": "a perfectly valid question", "k": 3})

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal_server_error"
    # The client-facing message must never contain the raw exception text/internals.
    assert "secret internal path" not in body["detail"]
    assert "/etc/x" not in body["detail"]
