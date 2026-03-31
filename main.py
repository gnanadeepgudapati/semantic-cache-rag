# main.py
from fastapi import FastAPI
from models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from config import settings
from ingest import chunk_text
from embed import embed_corpus, hash_embed
from search import retrieve
from bm25 import build_bm25, bm25_search
from fusion import reciprocal_rank_fusion
from openai import OpenAI

app = FastAPI(title="RAG Journey API")
client = OpenAI(api_key=settings.openai_api_key)

corpus = []

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(corpus)}

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    chunks = chunk_text(request.text, settings.chunk_size, settings.overlap)
    for i, chunk in enumerate(chunks):
        corpus.append({
            "id": f"{request.title}_chunk{i}",
            "source": request.title,
            "text": chunk,
            "embedding": hash_embed(chunk)
        })
    return IngestResponse(
        message=f"Successfully ingested {request.title}",
        chunks_created=len(chunks)
    )

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not corpus:
        return QueryResponse(answer="No documents ingested yet.", sources=[])

    vector_results = retrieve(request.question, corpus, settings.top_k)
    bm25_index = build_bm25(corpus)
    bm25_results = bm25_search(request.question, corpus, bm25_index, settings.top_k)
    results = reciprocal_rank_fusion(vector_results, bm25_results)[:settings.top_k]

    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the context provided below. If the answer isn't "
                    "in the context, say so.\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            {"role": "user", "content": request.question},
        ],
    )

    answer = response.choices[0].message.content
    sources = list(set(r["source"] for r in results))
    return QueryResponse(answer=answer, sources=sources)