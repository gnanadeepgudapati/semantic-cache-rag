# fusion.py

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    
    scores = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + k)

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (rank + k)

    all_chunks = {chunk["id"]: chunk for chunk in vector_results + bm25_results}

    fused = sorted(all_chunks.values(), key=lambda c: scores[c["id"]], reverse=True)

    return fused