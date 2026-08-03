"""
main.py — FastAPI service for the RAG Knowledge Extraction System.

Endpoints:
    GET  /health        liveness/readiness check
    POST /search         retrieval only (no LLM call) -- Week 2's semantic_search over HTTP
    POST /query           full RAG: retrieval + generation -- Week 3's rag_answer over HTTP

Run with:
    uvicorn main:app --reload
"""

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from starlette.datastructures import MutableHeaders

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from rag_pipeline import RAGNotReadyError, RAGPipeline

# ---------------------------------------------------------------------------
# Structured logging -- one JSON line per request, easy to grep/ingest
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "path", "method", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# App + pipeline
# ---------------------------------------------------------------------------
pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline.load()
        logger.info("Pipeline loaded at startup", extra={"path": "startup", "method": "INTERNAL"})
    except RAGNotReadyError as e:
        # Don't crash the process -- let /health report the failure so the API stays
        # inspectable rather than refusing to start entirely.
        logger.error("Pipeline failed to load at startup: %s", e,
                      extra={"path": "startup", "method": "INTERNAL"})
    yield


app = FastAPI(title="RAG Knowledge Extraction API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Structured request logging middleware
#
# Implemented as pure ASGI middleware (not the `@app.middleware("http")` /
# BaseHTTPMiddleware helper) deliberately: BaseHTTPMiddleware has a well-known Starlette
# interaction bug where exceptions raised inside the endpoint can bypass registered
# `@app.exception_handler(Exception)` handlers and propagate raw instead of becoming a
# clean 500 response. Pure ASGI middleware doesn't have that failure mode.
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        t0 = time.perf_counter()
        status_holder = {"status_code": None}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status_code"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(
                "request completed",
                extra={"request_id": request_id, "path": scope.get("path", ""),
                       "method": scope.get("method", ""),
                       "status_code": status_holder["status_code"], "duration_ms": duration_ms},
            )


app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer.")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    category: Optional[str] = Field(None, description="Optional ArXiv category filter, e.g. 'cs.CL'.")
    topic_id: Optional[int] = Field(None, description="Optional Week 4 topic_id filter.")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        # Pydantic's min_length=1 catches "" but not whitespace-only strings like "   " --
        # that business-rule check is deliberately done here, not silently accepted.
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return v


class SourceItem(BaseModel):
    rank: int
    score: float
    doc_id: str
    title: str
    category: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    backend: str
    sources: List[SourceItem]
    out_of_domain: bool
    possible_hallucination: bool
    groundedness_overlap: Optional[float]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    category: Optional[str] = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return v


class SearchResultItem(BaseModel):
    rank: int
    score: float
    doc_id: str
    title: str
    category: str
    text: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embedding_backend: Optional[str]


# ---------------------------------------------------------------------------
# Error handlers -- explicit 400 / 422 / 500 behavior
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Pydantic/FastAPI validation failures (wrong type, missing field, blank query, k out of
    # range, ...) -> 422, with the specific field errors included so a client can fix the request.
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("validation error", extra={"request_id": request_id, "path": request.url.path,
                                               "method": request.method, "status_code": 422})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "request_id": request_id,
                 "detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Anything not already an HTTPException -> 500. Logged with the full exception server-side,
    # but the client only ever sees a generic message -- no stack trace or internals leaked.
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("unhandled exception: %s", exc, exc_info=True,
                 extra={"request_id": request_id, "path": request.url.path,
                        "method": request.method, "status_code": 500})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "request_id": request_id,
                 "detail": "An unexpected error occurred. Please try again."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _validate_category(category: Optional[str]) -> None:
    """A category that doesn't exist in the corpus is a genuine 400: the request is
    well-formed JSON/schema-wise, but semantically invalid -- the client should pick a real one."""
    if category is None or pipeline.chunks_df is None:
        return
    known = set(pipeline.chunks_df["category"].unique().tolist())
    if category not in known:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown category '{category}'. Valid categories: {sorted(known)}",
        )


def _validate_topic_id(topic_id: Optional[int]) -> None:
    if topic_id is None or pipeline.chunks_df is None or "topic_id" not in pipeline.chunks_df.columns:
        return
    known = set(int(t) for t in pipeline.chunks_df["topic_id"].unique().tolist())
    if topic_id not in known:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown topic_id {topic_id}. Valid topic_ids: {sorted(known)}",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not pipeline.is_ready:
        # Service is up but not ready to serve real queries -- 503, not a silent 200.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Pipeline not loaded.")
    return HealthResponse(status="ok", chunks_indexed=pipeline.collection.count(),
                           embedding_backend=pipeline.embedding_backend)


@app.post("/search", response_model=List[SearchResultItem])
def search(payload: SearchRequest) -> List[SearchResultItem]:
    if not pipeline.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Pipeline not loaded.")
    _validate_category(payload.category)
    # No broad try/except here: an unexpected exception (a real bug, a dependency failure)
    # is deliberately left to propagate to the global handler below, which turns it into a
    # 500 -- a well-formed, semantically valid request failing unexpectedly IS a server error,
    # not a client error, and should be logged and reported as one rather than masked as a 400.
    results = pipeline.semantic_search(payload.query, k=payload.k, category=payload.category)
    return results.to_dict("records")


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    if not pipeline.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Pipeline not loaded.")
    _validate_category(payload.category)
    _validate_topic_id(payload.topic_id)
    result = pipeline.answer(payload.query, k=payload.k, category=payload.category,
                              topic_id=payload.topic_id)
    return result
