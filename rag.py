# rag.py
from openai import OpenAI
from dotenv import load_dotenv
from ingest import build_corpus
from embed import embed_corpus
from search import retrieve

load_dotenv()
client = OpenAI()


def ask(question: str, corpus: list[dict]) -> str:
    # 1. Find relevant chunks
    results = retrieve(question, corpus, top_k=3)

    # 2. Build context string from top chunks
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    # 3. Call OpenAI with question + context
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
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("Building corpus...")
    corpus = embed_corpus(build_corpus("docs"))
    print(f"Ready — {len(corpus)} chunks indexed.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() == "quit":
            break

        answer = ask(question, corpus)
        print(f"\nAnswer: {answer}\n")

        results = retrieve(question, corpus, top_k=3)
        print("Sources used:")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']}: {r['text'][:80]}...")
        print()