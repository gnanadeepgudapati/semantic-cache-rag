# main.py
import uuid
import time
from fastapi import FastAPI, HTTPException, Request
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

app = FastAPI(title="RAG Journey API")
client = OpenAI(api_key=settings.openai_api_key)
corpus = []

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    return response

@app.get("/health")
def health():
    log.info("health_check", chunks_indexed=len(corpus))
    return {"status": "ok", "chunks_indexed": len(corpus)}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, req: Request):
    request_id = req.state.request_id
    log.info("ingest_start", request_id=request_id, title=request.title)

    chunks = chunk_text(request.text, settings.chunk_size, settings.overlap)
    for i, chunk in enumerate(chunks):
        corpus.append({
            "id": f"{request.title}_chunk{i}",
            "source": request.title,
            "text": chunk,
            "embedding": hash_embed(chunk)
        })

    log.info("ingest_complete", request_id=request_id, title=request.title, chunks_created=len(chunks))
    return IngestResponse(message=f"Successfully ingested {request.title}", chunks_created=len(chunks))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, req: Request):
    request_id = req.state.request_id
    log.info("query_start", request_id=request_id, question=request.question)

    if not corpus:
        return QueryResponse(answer="No documents ingested yet.", sources=[], confidence=0.0)

    t0 = time.time()
    vector_results = retrieve(request.question, corpus, settings.top_k)
    bm25_index = build_bm25(corpus)
    bm25_results = bm25_search(request.question, corpus, bm25_index, settings.top_k)
    results = reciprocal_rank_fusion(vector_results, bm25_results)[:settings.top_k]
    retrieval_ms = int((time.time() - t0) * 1000)

    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(request.question, context)}
    ]

    t1 = time.time()
    try:
        structured = get_structured_response(client, messages)
    except Exception as e:
        log.error("query_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    llm_ms = int((time.time() - t1) * 1000)

    log.info("query_complete",
        request_id=request_id,
        question=request.question,
        chunks_retrieved=len(results),
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        confidence=structured.confidence
    )

    return QueryResponse(answer=structured.answer, sources=structured.sources, confidence=structured.confidence)