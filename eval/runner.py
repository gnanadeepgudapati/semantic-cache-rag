# eval/runner.py
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import chunk_text
from embed import hash_embed
from search import retrieve
from bm25 import build_bm25, bm25_search
from fusion import reciprocal_rank_fusion
from response_parser import get_structured_response
from Prompt import SYSTEM_PROMPT, build_user_prompt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

DOCS = [
    {
        "title": "python_comprehensive",
        "text": "Python is a high-level, general-purpose programming language created by Guido van Rossum. Development began in the late 1980s and the first version was released in 1991. Van Rossum named the language after the British comedy group Monty Python. Python was designed with an emphasis on code readability and simplicity. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python 2 was released in 2000. Python 3 was released in 2008 and was a major revision not backward compatible with Python 2. Python 2 reached end of life in January 2020. Python is widely used in web development with frameworks like Django and Flask. Major companies that use Python include Google, Facebook, Instagram, Spotify, Netflix, and Dropbox. Python uses dynamic typing meaning variable types are determined at runtime. The Python Package Index known as PyPI hosts over 400000 packages. The Zen of Python written by Tim Peters describes the philosophy of Python design."
    }
]

def build_corpus():
    corpus = []
    for doc in DOCS:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            corpus.append({
                "id": f"{doc['title']}_chunk{i}",
                "source": doc["title"],
                "text": chunk,
                "embedding": hash_embed(chunk)
            })
    return corpus

def answer_question(question: str, corpus: list) -> dict:
    vector_results = retrieve(question, corpus, top_k=3)
    bm25_index = build_bm25(corpus)
    bm25_results = bm25_search(question, corpus, bm25_index, top_k=3)
    results = reciprocal_rank_fusion(vector_results, bm25_results)[:3]

    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, context)}
    ]

    structured = get_structured_response(client, messages)
    return {
        "answer": structured.answer,
        "confidence": structured.confidence
    }

def score_case(answer: str, expected: str) -> bool:
    return expected.lower() in answer.lower()

def run_eval():
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    with open(dataset_path) as f:
        dataset = json.load(f)

    corpus = build_corpus()
    results = []

    print(f"\nRunning eval on {len(dataset)} questions...\n")

    for case in dataset:
        result = answer_question(case["question"], corpus)
        passed = score_case(result["answer"], case["expected"])
        results.append({
            "question": case["question"],
            "expected": case["expected"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "passed": passed
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['question']}")
        print(f"       expected: {case['expected']}")
        print(f"       got: {result['answer'][:80]}...")
        print()

    pass_rate = sum(r["passed"] for r in results) / len(results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    print(f"pass_rate: {pass_rate:.2f}")
    print(f"avg_confidence: {avg_confidence:.2f}")

    return pass_rate, avg_confidence