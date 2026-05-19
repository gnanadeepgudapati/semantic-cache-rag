# RAG with Cache Memory

> 🚧 **Work in progress** — adding a semantic cache layer on top of a production-grade RAG pipeline so that repeated or similar questions are answered instantly from cache instead of going through the full retrieval + LLM round-trip.

---

## The Idea

Every time a user asks a question, a standard RAG system does the same expensive work — embed the query, retrieve chunks, call the LLM. If someone asks a question that's semantically close to one already answered, why pay that cost again?

This project adds a **semantic cache** in front of the RAG pipeline:

1. Incoming query is embedded.
2. The embedding is compared against a cache of previously answered queries.
3. If a similar-enough query exists in cache (above a configurable similarity threshold), the cached answer is returned immediately — no retrieval, no LLM call.
4. If no cache hit, the full RAG pipeline runs and the result is stored in cache for future use.

```
User Query
    │
    ▼
[ Semantic Cache ]──── cache hit ────▶ Return cached answer
    │
  cache miss
    │
    ▼
[ Hybrid Retrieval (BM25 + Vector + RRF) ]
    │
    ▼
[ LLM (OpenAI) ]
    │
    ▼
[ Store in Cache ] ──▶ Return answer
```

---

## Current Architecture

```
docs/              → source documents
ingest.py          → document loading and chunking
embed.py           → query & document embeddings
search.py          → cosine similarity vector search
bm25.py            → keyword search
fusion.py          → reciprocal rank fusion (combines vector + keyword)
rag_prompt.py      → prompt templates
response_parser.py → structured output parsing with retry
main.py            → FastAPI app with rate limiting and circuit breaker
logger.py          → structured JSON logging
config.py          → settings from .env
models.py          → Pydantic request/response models
eval/              → evaluation framework
cache.py           → 🚧 semantic cache (in progress)
```

---

## Stages Built

- Stage 1 — Toy RAG CLI
- Stage 2 — FastAPI wrapper
- Stage 3 — Hybrid retrieval (BM25 + vector + RRF)
- Stage 4 — Structured extraction with confidence scores
- Stage 5 — Evaluation framework
- Stage 6 — Structured logging with request IDs
- Stage 7 — Production hardening (rate limiting, circuit breaker)
- Stage 8 — Docker containerization
- Stage 9 — 🚧 Semantic cache layer *(in progress)*

---

## Roadmap for Stage 9 — Semantic Cache

- [ ] `cache.py` — in-memory cache with cosine similarity lookup
- [ ] Configurable similarity threshold (default `0.92`)
- [ ] Cache TTL / max size with LRU eviction
- [ ] Cache hit/miss logged as structured events
- [ ] `/cache/stats` endpoint — hit rate, size, top queries
- [ ] `/cache/clear` admin endpoint
- [ ] Eval suite extended to measure cache hit rate and latency savings

---

## Setup

### Option 1 — Docker (recommended)

```bash
# clone the repo, add your key
echo "OPENAI_API_KEY=your-key-here" > .env
docker compose up --build
```
Open http://localhost:8000/docs

### Option 2 — Local

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
uvicorn main:app --reload
```
Open http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET | /health | Server status | None |
| POST | /ingest | Upload a document | 20/minute |
| POST | /query | Ask a question (cache-aware) | 10/minute |
| GET | /cache/stats | 🚧 Cache hit rate & stats | None |
| DELETE | /cache/clear | 🚧 Flush the cache | None |

---

## Evaluation

```bash
python -m eval
```

Expected output (current):
```
pass_rate: 1.00
avg_confidence: 0.96
```

Stage 9 target:
```
pass_rate:      1.00
avg_confidence: 0.96
cache_hit_rate: ~0.40   (on repeated / paraphrased queries)
avg_latency_ms: <50     (cache hits vs ~800ms full pipeline)
```