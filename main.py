# main.py
import uuid
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from config import settings
from ingest import chunk_text
from embed import hash_embed
from search import retrieve
from bm25 import build_bm25, bm25_search
from fusion import reciprocal_rank_fusion
from rag_prompt import SYSTEM_PROMPT, build_user_prompt
from response_parser import get_structured_response
from logger import setup_logging, get_logger
from openai import OpenAI

setup_logging()
log = get_logger()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Journey API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = OpenAI(api_key=settings.openai_api_key)
corpus = []

openai_failures = 0
CIRCUIT_BREAKER_THRESHOLD = 3
circuit_open = False

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    return response

@app.get("/health")
def health():
    log.info("health_check", chunks_indexed=len(corpus), circuit_open=circuit_open)
    return {"status": "ok", "chunks_indexed": len(corpus), "circuit_open": circuit_open}

@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("20/minute")
async def ingest(request: Request, body: IngestRequest):
    request_id = request.state.request_id

    if not body.title.strip():
        raise HTTPException(status_code=422, detail="title cannot be empty")
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")
    if len(body.text) > 100000:
        raise HTTPException(status_code=422, detail="text too long, max 100000 characters")

    log.info("ingest_start", request_id=request_id, title=body.title)
    chunks = chunk_text(body.text, settings.chunk_size, settings.overlap)
    for i, chunk in enumerate(chunks):
        corpus.append({
            "id": f"{body.title}_chunk{i}",
            "source": body.title,
            "text": chunk,
            "embedding": hash_embed(chunk)
        })
    log.info("ingest_complete", request_id=request_id, title=body.title, chunks_created=len(chunks))
    return IngestResponse(message=f"Successfully ingested {body.title}", chunks_created=len(chunks))


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    global openai_failures, circuit_open
    request_id = request.state.request_id

    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty")
    if len(body.question) > 1000:
        raise HTTPException(status_code=422, detail="question too long, max 1000 characters")

    if circuit_open:
        log.warning("circuit_open", request_id=request_id)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    if not corpus:
        return QueryResponse(answer="No documents ingested yet.", sources=[], confidence=0.0)

    t0 = time.time()
    vector_results = retrieve(body.question, corpus, settings.top_k)
    bm25_index = build_bm25(corpus)
    bm25_results = bm25_search(body.question, corpus, bm25_index, settings.top_k)
    results = reciprocal_rank_fusion(vector_results, bm25_results)[:settings.top_k]
    retrieval_ms = int((time.time() - t0) * 1000)

    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(body.question, context)}
    ]

    t1 = time.time()
    try:
        structured = get_structured_response(client, messages)
        openai_failures = 0
        circuit_open = False
    except Exception as e:
        openai_failures += 1
        log.error("openai_failure", request_id=request_id, failures=openai_failures, error=str(e))
        if openai_failures >= CIRCUIT_BREAKER_THRESHOLD:
            circuit_open = True
            log.error("circuit_breaker_opened", failures=openai_failures)
        raise HTTPException(status_code=500, detail=str(e))
    llm_ms = int((time.time() - t1) * 1000)

    log.info("query_complete",
        request_id=request_id,
        question=body.question,
        chunks_retrieved=len(results),
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        confidence=structured.confidence
    )
    return QueryResponse(answer=structured.answer, sources=structured.sources, confidence=structured.confidence)