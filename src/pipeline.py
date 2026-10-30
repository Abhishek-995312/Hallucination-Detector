"""Phase 5b - The full pipeline.

question -> retrieve -> LLM answer -> claims -> verify each claim -> scores

Run from the project root:
    python -m src.pipeline "What does the liver do?"                # RAG answer
    python -m src.pipeline "What does the liver do?" --plain        # LLM answer without evidence
    python -m src.pipeline "Any question" --answer "Text to check"  # check an answer you provide
"""

import argparse

from src.claims import extract_claims
from src.generate import generate_answer, generate_without_evidence
from src.retrieve import Retriever
from src.score import summarize
from src.verify import CONTRADICTED, SUPPORTED, Verifier

ICON = {SUPPORTED: "[OK]  ", CONTRADICTED: "[FAIL]", "Unsupported": "[?]   "}


class HallucinationDetector:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.verifier = Verifier(self.retriever)  # shares the same embedding model

    def run(self, question: str, answer: str | None = None, use_rag: bool = True) -> dict:
        if answer is None:
            if use_rag:
                hits = self.retriever.search(question, k=5)
                answer = generate_answer(question, hits)
            else:
                answer = generate_without_evidence(question)

        claims = extract_claims(answer)
        results = self.verifier.verify_claims(claims)
        return {"question": question, "answer": answer,
                "claims": results, "summary": summarize(results)}


def print_report(out: dict) -> None:
    print(f"\nQUESTION: {out['question']}")
    print(f"\nANSWER:\n{out['answer']}\n")
    print("CLAIMS:")
    for i, r in enumerate(out["claims"], 1):
        print(f"\n{i}. {ICON[r['label']]} {r['label']} (conf {r['confidence']}, "
              f"relevance {r['relevance']})")
        print(f"   Claim:    {r['claim']}")
        if r["evidence"]:
            print(f"   Evidence: [{r['source']}] {r['evidence'][:220]}...")

    s = out["summary"]
    print("\n" + "=" * 60)
    print(f"VERDICT: {s['verdict']}")
    if s["n_claims"]:
        print(f"Claims: {s['n_claims']}  |  Supported {s['supported']}  |  "
              f"Contradicted {s['contradicted']}  |  Unsupported {s['unsupported']}")
        print(f"Grounding score: {s['grounding_score']}  "
              f"(confidence-weighted: {s['weighted_grounding']})")
        print(f"Contradiction rate: {s['contradiction_rate']}  |  "
              f"Unsupported rate: {s['unsupported_rate']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--plain", action="store_true", help="LLM answers without retrieved evidence")
    ap.add_argument("--answer", help="check this answer instead of generating one")
    args = ap.parse_args()

    detector = HallucinationDetector()
    print_report(detector.run(args.question, answer=args.answer, use_rag=not args.plain))