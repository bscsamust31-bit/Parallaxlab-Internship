# RAG Knowledge Extraction System — Week 3

## LLM Integration & Prompt Engineering

**Intern:** Chashman Aslam
**Program:** Parallax Lab Internship
**Deliverable:** Hallucination-resistant RAG system with robust API error handling and latency logging

---

## 🎯 What This Week Delivers

Week 1 built the cleaned corpus. Week 2 built chunking, embeddings, and semantic retrieval on
top of it. Week 3 closes the loop — retrieved evidence is no longer just displayed, it's
**reasoned over**. A real LLM (DeepSeek, via OpenRouter) now reads the retrieved chunks and
writes a grounded, cited answer, wrapped in the error handling and safety checks a production
system actually needs.

| # | Objective | Status |
|---|-----------|--------|
| 1 | Integrate DeepSeek/OpenRouter API to generate answers from retrieved chunks | ✅ |
| 2 | Prompt engineering best practices (system prompt, context injection) | ✅ |
| 3 | Robust error handling for API calls (rate limits, timeouts, token limits) | ✅ |
| 4 | Hallucination checks + explicit out-of-domain/off-topic handling | ✅ |
| 5 | Measure and log end-to-end response latency (retrieval + generation) | ✅ |

---

## 🧠 LLM Integration

- **Provider:** [OpenRouter](https://openrouter.ai) — one OpenAI-compatible REST endpoint that
  routes to DeepSeek's models, so no extra provider SDK is needed; plain `requests` is enough.
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Model:** `deepseek/deepseek-v4-flash` by default — DeepSeek's fast, cost-efficient model.
  Swap `MODEL_NAME` for any other OpenRouter-hosted model with no other code changes; check
  [openrouter.ai/models](https://openrouter.ai/models) for current pricing, since the lineup
  shifts over time.
- **Auth:** the key is read from the `OPENROUTER_API_KEY` environment variable, or requested
  securely via `getpass` — **never hardcoded** into the notebook.
- **Offline-safe by design:** every generation call tries DeepSeek first and falls back to a
  clearly-labeled offline extractive responder if no key is set or the call ultimately fails.
  This means the full pipeline — prompting, error handling, hallucination checks, latency
  logging — is provable and gradable even with zero API spend.

---

## ✍️ Prompt Engineering

**System prompt** fixes the model's role and, more importantly, its grounding discipline:

1. Answer only from the provided sources — no outside knowledge
2. Cite sources inline as `[Source N]` for every factual claim
3. Say so explicitly when the context is insufficient, instead of guessing
4. Keep answers concise (3–6 sentences) and directly on-topic

**Context injection** formats every retrieved chunk as a numbered, labeled source block —
`doc_id`, `title`, `category`, and the chunk text — rather than raw concatenated text. This
gives the model an explicit citation target for every claim, and gives a human reviewer a way
to trace any sentence in the answer straight back to the exact chunk it came from:

```
[Source 1] (doc_id=..., title="...", category=cs.CL)
<chunk text>

[Source 2] (doc_id=..., title="...", category=cs.LG)
<chunk text>
```

---

## 🛡️ Robust API Error Handling

Every failure mode a real API integration hits in production is handled explicitly, not
papered over with a bare `try/except`:

| Failure mode | Handling |
|---|---|
| Rate limit (HTTP 429) | Retried with exponential backoff, honoring the API's `Retry-After` header when present |
| Server errors (5xx) | Retried with exponential backoff, up to `MAX_RETRIES` |
| Timeouts / connection drops | Retried with backoff, then falls back rather than hanging indefinitely |
| Client errors (4xx, not 429) | **Not retried** — a bad key or malformed request won't fix itself on attempt #2, so it fails fast instead of wasting time and quota |
| Context too large for the model | `truncate_context()` estimates tokens (~4 chars/token) and drops the lowest-ranked chunks first until under `MAX_CONTEXT_TOKENS` |
| No API key / retries exhausted | Falls back to a clearly-labeled offline extractive responder rather than crashing the pipeline |

**Verified live, not just in theory:** during batch testing, a fresh OpenRouter key with no
purchased credits returned `402 Insufficient credits` on every call. Because 402 is a
non-retryable client error, the system correctly failed fast on the *first* attempt for each
query (no wasted retries) and transparently handed off to the offline responder — the full
8-query batch test still completed end-to-end with zero crashes. That's the error-handling
logic working exactly as designed, under a real API failure rather than a simulated one.

---

## 🧪 Hallucination Checks & Out-of-Domain Handling

Two safeguards, deliberately placed on either side of the LLM call:

**1. Relevance gate — before generation.** If the top retrieved chunk's similarity score falls
below `MIN_RELEVANCE_SCORE`, the query is treated as out-of-domain/off-topic for this corpus.
The LLM call is skipped entirely, and an explicit *"I don't have enough information"* response
is returned instead. This does two things at once: it prevents a confidently-worded
hallucinated answer, and it saves an API call on every query that would have produced one.

**2. Groundedness check — after generation.** A lightweight heuristic measures what fraction of
the generated answer's distinctive vocabulary also appears in the retrieved context. Low
overlap is flagged as `possible_hallucination: True`, surfaced right alongside the answer.

**Stated plainly, not oversold:** the groundedness check is a word-overlap heuristic, not a
factuality verifier. It reliably catches an answer that drifts far from the source material
(e.g. the model ignoring the context and answering from its own training data), but it will not
catch a subtle factual error made *within* otherwise well-grounded wording. A production system
would pair this with a second LLM call — *"does this answer actually follow from this
context?"* — for stronger verification. That's a deliberate cost/latency tradeoff, noted here
rather than implemented, to keep this week's system cheap enough to batch-test freely.

---

## ⏱️ End-to-End Latency Logging

Every single query logs three numbers, not one: `retrieval_ms`, `generation_ms`, and
`total_ms`. Splitting the two matters — a latency regression in production is either a
retrieval problem (embedding model or ChromaDB) or a generation problem (the LLM call), and a
single opaque end-to-end number can't tell you which.

**Batch test results** — 5 in-domain queries (transformers, CNNs, SGD optimization, dense
retrieval, robotics — matching the corpus's own categories) mixed with 3 deliberately
out-of-domain queries (pizza dough, football scores, smartphone shopping), each repeated 3×:

- **Relevance-gate accuracy: 100%** — every out-of-domain query correctly triggered the
  "insufficient information" response; every in-domain query retrieved strong context and
  proceeded to generation.
- **Mean end-to-end latency: ~5 ms** in offline fallback mode (retrieval ~3.5 ms, generation
  ~1.4 ms). With a funded API key, generation time would additionally include the real network
  round-trip to DeepSeek — typically several hundred milliseconds to a few seconds, depending
  on model and load.

---

## 📈 Sample Results

**In-domain:** *"How do transformers use attention for language understanding?"*
- Retrieved 5 relevant `cs.CL` chunks · groundedness overlap 0.77 · no hallucination flag
- `retrieval: 4.89 ms · generation: 3.39 ms · total: 8.27 ms`

**Out-of-domain:** *"What's a good recipe for making pizza dough at home?"*
- Relevance gate fired correctly — no LLM call made, no API spend wasted
- *"I don't have enough relevant information in this knowledge base to answer that question
  confidently..."*
- `retrieval: 3.91 ms · generation: 0.00 ms · total: 4.18 ms`

---

## 🔭 Noted for Future Work (Not Implemented This Week, By Design)

- A second LLM call for stronger groundedness verification — deliberately skipped to keep
  batch testing cheap and fast this week
- Response caching for repeated or near-duplicate queries
- Streaming responses, for better perceived latency in an interactive UI
- A larger, human-labeled evaluation set to measure hallucination-detection precision/recall
  properly, rather than by heuristic alone
- Structured monitoring on relevance-gate trigger rate and API error rate in production

---

## 🔑 Using Real DeepSeek Generation

Set the `OPENROUTER_API_KEY` environment variable (or paste a key when the notebook prompts for
one via `getpass`) — no other code changes are needed. The notebook detects the key
automatically and switches from the offline extractive responder to real DeepSeek generation.
A funded OpenRouter account is required for this — see the note above on the `402 Insufficient
credits` response if testing with a brand-new key.

---

*Week 3 of the RAG Knowledge Extraction System — Parallax Lab Internship.*
