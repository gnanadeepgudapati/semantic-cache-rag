# ingest.py
from pathlib import Path

def load_documents(folder: str) -> list[dict]:
    """Read all .txt and .md files from a folder."""
    docs = []
    for path in Path(folder).glob("*"):
        if path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
            docs.append({"title": path.name, "text": text})
    return docs
def chunk_text(text: str, chunk_size: int = 100, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks



def build_corpus(folder: str) -> list[dict]:
    """Load docs, chunk them, return a flat list of chunk records."""
    corpus = []
    for doc in load_documents(folder):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            corpus.append({
                "id": f"{doc['title']}_chunk{i}",
                "source": doc["title"],
                "text": chunk,
            })
    return corpus    

if __name__ == "__main__":
    corpus = build_corpus("docs")
    print(f"Total chunks: {len(corpus)}")
    print(f"\nFirst chunk:")
    print(corpus[0])
    