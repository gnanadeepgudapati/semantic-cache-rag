# bm25.py
from rank_bm25 import BM25Okapi

def build_bm25(corpus: list[dict]) -> BM25Okapi:
    """Build a BM25 index from the corpus."""
    tokenized = [chunk["text"].lower().split() for chunk in corpus]
    return BM25Okapi(tokenized)

def bm25_search(query: str, corpus: list[dict], bm25: BM25Okapi, top_k: int = 3) -> list[dict]:
    """Score every chunk using BM25 keyword matching."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = []
    for i, chunk in enumerate(corpus):
        scored.append({**chunk, "score": float(scores[i])})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]