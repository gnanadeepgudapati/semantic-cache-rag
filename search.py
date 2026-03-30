# search.py
import numpy as np
from embed import hash_embed

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measure how similar two vectors are.
    Returns a number between -1.0 and 1.0.
    """
    return float(np.dot(a, b))

def retrieve(query: str, corpus: list[dict], top_k: int = 3) -> list[dict]:
    """
    Embed the query, score every chunk, return top_k matches.
    """
    query_vec = hash_embed(query)

    scored = []
    for chunk in corpus:
        score = cosine_similarity(query_vec, chunk["embedding"])
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

if __name__ == "__main__":
    from ingest import build_corpus
    from embed import embed_corpus

    corpus = embed_corpus(build_corpus("docs"))
    
    query = "Who created Python?"
    results = retrieve(query, corpus, top_k=3)
    
    print(f"Query: {query}\n")
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['text'][:80]}...")