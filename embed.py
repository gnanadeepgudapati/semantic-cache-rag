# embed.py
import hashlib
import numpy as np

VECTOR_DIM = 128

def hash_embed(text: str) -> np.ndarray:
    vec = np.zeros(VECTOR_DIM)
    words = text.lower().split()

    for word in words:
        digest = hashlib.md5(word.encode()).digest()
        index = int.from_bytes(digest[:2], "big") % VECTOR_DIM
        sign = 1 if digest[2] % 2 == 0 else -1
        vec[index] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
def embed_corpus(corpus: list[dict]) -> list[dict]:
    """Add an embedding to each chunk in the corpus."""
    for chunk in corpus:
        chunk["embedding"] = hash_embed(chunk["text"])
    return corpus

if __name__ == "__main__":
    from ingest import build_corpus
    corpus = build_corpus("docs")
    corpus = embed_corpus(corpus)
    print(f"Total chunks: {len(corpus)}")
    print(f"Keys in first chunk: {corpus[0].keys()}")
    print(f"Embedding length: {len(corpus[0]['embedding'])}")