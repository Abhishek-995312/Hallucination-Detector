"""Phase 4 - Claim verification.

For each claim:
  1. retrieve the top-K evidence chunks for THAT claim
  2. cut each chunk into short sentence windows (NLI works good for short text)
  3. drop windows that are not about the same topic as the claim (relevance filter)
  4. run NLI on (evidence window, claim) -> entailment / contradiction / neutral
  5. turn the scores into Supported / Contradicted / Unsupported

Quick test from the project root:
    python -m src.verify
    python -m src.verify "The heart has two chambers."
"""

import re
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.retrieve import Retriever

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
TOP_K = 5               # evidence chunks retrieved per claim
WINDOW_SENTENCES = 3    # sentences per evidence window
WINDOW_STRIDE = 2       # windows overlap by one sentence
ENTAIL_T = 0.70         # min entailment probability to call a claim Supported
CONTRA_T = 0.70         # min contradiction probability to call a claim Contradicted
CONTRA_MARGIN = 0.0     # extra margin contradiction needs over entailment (0 = just higher)
MIN_WINDOW_SIM = 0.40   # evidence windows less similar than this to the claim are ignored
BATCH_SIZE = 16

SUPPORTED, CONTRADICTED, UNSUPPORTED = "Supported", "Contradicted", "Unsupported"


def make_windows(text: str) -> list[str]:
    """Split a chunk into overlapping windows of a few sentences."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sents) <= WINDOW_SENTENCES:
        return [text]
    windows, i = [], 0
    while True:
        windows.append(" ".join(sents[i:i + WINDOW_SENTENCES]))
        if i + WINDOW_SENTENCES >= len(sents):
            break
        i += WINDOW_STRIDE
    return windows


def decide(max_ent: float, max_con: float) -> str:
    """Three-way decision from the best entailment / contradiction scores."""
    if max_ent >= ENTAIL_T and max_ent >= max_con:
        return SUPPORTED
    if max_con >= CONTRA_T and max_con - max_ent >= CONTRA_MARGIN:
        return CONTRADICTED
    return UNSUPPORTED


class Verifier:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()
        self.embedder = self.retriever.model  # same embedding model, reused for relevance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.to(self.device).eval()

        # read label order from the model instead of assuming it
        labels = {l.lower(): int(i) for i, l in self.model.config.id2label.items()}
        self.i_ent = labels["entailment"]
        self.i_con = labels["contradiction"]

    @torch.no_grad()
    def _nli_probs(self, premises: list[str], claim: str) -> torch.Tensor:
        """Softmax probabilities, shape [len(premises), 3]."""
        out = []
        for s in range(0, len(premises), BATCH_SIZE):
            batch = premises[s:s + BATCH_SIZE]
            enc = self.tokenizer(
                batch, [claim] * len(batch), truncation="only_first",
                max_length=512, padding=True, return_tensors="pt",
            ).to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu())
        return torch.cat(out)

    def verify_claim(self, claim: str) -> dict:
        hits = self.retriever.search(claim, k=TOP_K)
        candidates = [(w, h) for h in hits for w in make_windows(h["text"])]

        result = {"claim": claim, "label": UNSUPPORTED, "confidence": 0.0,
                  "evidence": None, "source": None, "url": None,
                  "entail": 0.0, "contradict": 0.0, "relevance": 0.0,
                  "contradict_evidence": None, "contradict_source": None}
        if not candidates:
            return result

        # Relevance filter: NLI models can output "contradiction" for text that
        # is simply about something else. Only judge windows on the claim's topic.
        c_emb = self.embedder.encode([claim], normalize_embeddings=True)
        w_emb = self.embedder.encode([w for w, _ in candidates],
                                     normalize_embeddings=True, batch_size=32)
        sims = (w_emb @ c_emb.T).ravel()
        keep = [i for i, sim in enumerate(sims) if sim >= MIN_WINDOW_SIM]

        if not keep:  # nothing in the knowledge base is about this claim
            i = int(sims.argmax())
            window, hit = candidates[i]
            result.update(confidence=round(1.0 - float(sims[i]), 3),
                          evidence=window, source=hit["source"], url=hit["url"],
                          relevance=round(float(sims[i]), 3))
            return result

        kept = [candidates[i] for i in keep]
        kept_sims = [float(sims[i]) for i in keep]

        probs = self._nli_probs([w for w, _ in kept], claim)
        ent, con = probs[:, self.i_ent], probs[:, self.i_con]
        best_ent, best_con = int(ent.argmax()), int(con.argmax())

        label = decide(float(ent[best_ent]), float(con[best_con]))

        # evidence shown to the user: the window that decided the label.
        # For Unsupported, show the closest window (highest entailment).
        pick = best_con if label == CONTRADICTED else best_ent
        confidence = float(con[pick]) if label == CONTRADICTED else float(ent[pick])
        if label == UNSUPPORTED:
            confidence = 1.0 - max(float(ent[best_ent]), float(con[best_con]))

        window, hit = kept[pick]
        result.update(
            label=label, confidence=round(confidence, 3),
            evidence=window, source=hit["source"], url=hit["url"],
            entail=round(float(ent[best_ent]), 3),
            contradict=round(float(con[best_con]), 3),
            relevance=round(kept_sims[pick], 3),
            contradict_evidence=kept[best_con][0],
            contradict_source=kept[best_con][1]["source"],
        )
        return result

    def verify_claims(self, claims: list[str]) -> list[dict]:
        return [self.verify_claim(c) for c in claims]


if __name__ == "__main__":
    demo = [
        "The liver produces bile.",                 # expect Supported
        "The human heart has two chambers.",        # expect Contradicted
        "The Eiffel Tower is located in Paris.",    # expect Unsupported (not in KB)
    ]
    claims = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else demo

    verifier = Verifier()
    for r in verifier.verify_claims(claims):
        print(f"\nClaim:    {r['claim']}")
        print(f"Label:    {r['label']}  (confidence {r['confidence']})")
        print(f"Scores:   entail={r['entail']}  contradict={r['contradict']}  "
              f"relevance={r['relevance']}")
        if r["evidence"]:
            print(f"Source:   {r['source']}")
            print(f"Evidence: {r['evidence'][:300]}...")
        if r.get("contradict", 0) >= 0.5:
            print(f"Highest-contradiction window ({r['contradict_source']}): "
                  f"{r['contradict_evidence'][:300]}...")