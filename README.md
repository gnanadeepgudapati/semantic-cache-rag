# RAG Journey

A production-grade Retrieval Augmented Generation (RAG) system built from scratch.

## Architecture

```
docs/              → source documents
ingest.py          → document loading and chunking
embed.py           → hash embeddings (swap for OpenAI embeddings in production)
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
```

## Setup

### Option 1 — Docker (recommended)

1. Clone the repo
2. Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```
3. Run:
```bash
docker compose up --build
```
4. Open http://localhost:8000/docs

### Option 2 — Local

1. Clone the repo
2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```
5. Run:
```bash
uvicorn main:app --reload
```
6. Open http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| GET    | /health  | Server status | None |
| POST   | /ingest  | Upload a document | 20/minute |
| POST   | /query   | Ask a question | 10/minute |

## Evaluation

Run the eval suite:
```bash
python -m eval
```

Expected output:
```
pass_rate: 1.00
avg_confidence: 0.96
```

## Stages Built

- Stage 1 — Toy RAG CLI
- Stage 2 — FastAPI wrapper
- Stage 3 — Hybrid retrieval (BM25 + vector + RRF)
- Stage 4 — Structured extraction with confidence scores
- Stage 5 — Evaluation framework
- Stage 6 — Structured logging with request IDs
- Stage 7 — Production hardening (rate limiting, circuit breaker)
- Stage 8 — Docker containerization