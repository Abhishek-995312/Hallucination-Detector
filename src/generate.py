"""Phase 2 - Generate an answer from retrieved evidence.

Quick test from the project root:
    python -m src.generate "What does the liver do?"
"""

import sys

from src.llm import call_llm
from src.retrieve import Retriever

SYSTEM_WITH_EVIDENCE = (
    "You answer questions using ONLY the evidence passages provided. "
    "Write a clear answer in 2-5 sentences. "
    "Do not include citations, source numbers, or a sources list in the answer. "
    "If the evidence does not contain the answer, say you do not have enough information."
)

SYSTEM_NO_EVIDENCE = (
    "You answer questions in 2-5 sentences using your own knowledge."
)


def format_evidence(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )


def generate_answer(question: str, hits: list[dict]) -> str:
    """RAG answer: the LLM sees the question plus retrieved passages."""
    prompt = f"Evidence:\n{format_evidence(hits)}\n\nQuestion: {question}\n\nAnswer:"
    return call_llm(prompt, system=SYSTEM_WITH_EVIDENCE)


def generate_without_evidence(question: str) -> str:
    """Plain LLM answer with no retrieval. Useful for creating test data,
    since it hallucinates more often than the RAG answer."""
    return call_llm(f"Question: {question}\n\nAnswer:", system=SYSTEM_NO_EVIDENCE,
                    temperature=0.7)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What does the liver do?"
    hits = Retriever().search(question, k=5)

    print(f"Question: {question}\n")
    print("--- RAG answer (with evidence) ---")
    print(generate_answer(question, hits))
    print("\n--- Plain answer (no evidence) ---")
    print(generate_without_evidence(question))
    print("\n--- Sources used ---")
    for h in hits:
        print(f"- {h['source']} (chunk {h['chunk_index']}, score {h['score']})")