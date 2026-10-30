"""Phase 5a - Answer-level scores from claim-level results.

Unsupported claims are reported separately from Contradicted ones:
"not found in the knowledge base" is not the same as "false".
"""

from src.verify import CONTRADICTED, SUPPORTED, UNSUPPORTED


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_claims": 0, "supported": 0, "contradicted": 0, "unsupported": 0,
                "grounding_score": None, "weighted_grounding": None,
                "contradiction_rate": None, "unsupported_rate": None,
                "verdict": "No verifiable claims in the answer"}

    sup = [r for r in results if r["label"] == SUPPORTED]
    con = [r for r in results if r["label"] == CONTRADICTED]
    uns = [r for r in results if r["label"] == UNSUPPORTED]

    if con:
        verdict = "Contains claims contradicted by the evidence"
    elif uns:
        verdict = "Partly grounded: some claims are not backed by the knowledge base"
    else:
        verdict = "Fully grounded in the knowledge base"

    return {
        "n_claims": n,
        "supported": len(sup),
        "contradicted": len(con),
        "unsupported": len(uns),
        "grounding_score": round(len(sup) / n, 3),                       # share of Supported claims
        "weighted_grounding": round(sum(r["confidence"] for r in sup) / n, 3),  # weighted by confidence
        "contradiction_rate": round(len(con) / n, 3),
        "unsupported_rate": round(len(uns) / n, 3),
        "verdict": verdict,
    }
