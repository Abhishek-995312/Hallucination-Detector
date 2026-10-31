"""Phase 3 - Claim extraction.(convert large para into atomic claims).

Turn an LLM answer into short, single-fact, self-contained claims.

Quick test from the project root:
    python -m src.claims
    python -m src.claims "The liver filters blood. It is located in the abdomen."
"""

import json
import re
import sys

from src.llm import call_llm

SYSTEM = (
    "You extract atomic factual claims from a text. Follow these rules strictly:\n"
    "1. Each claim states exactly ONE verifiable fact.\n"
    "2. Each claim must be understandable on its own: replace pronouns "
    "(it, they, this) with the actual subject.\n"
    "3. Use only information that is in the text. Do not add, correct, or guess facts.\n"
    "4. Skip opinions, greetings, and statements like 'I do not have enough information'.\n"
    "5. Output ONLY a JSON list of strings, with no other text."
)

EXAMPLE = (
    "Text: The heart pumps blood and has four chambers.\n"
    'Output: ["The heart pumps blood.", "The heart has four chambers."]'
)

# Answers like this contain no factual claim to verify.
NO_INFO_PATTERNS = ("not have enough information", "do not have enough information",
                    "does not contain", "cannot answer")


def clean_answer(text: str) -> str:
    """Strip things that are not part of the answer: headers, question lines,
    '(Sources [1], [4])' and [1]-style citation markers."""
    text = re.sub(r"^-{2,}.*?-{2,}\s*$", "", text, flags=re.MULTILINE)  # --- headers ---
    text = re.sub(r"^Question:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\(\s*Sources?[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("\u2011", "-")  # odd hyphen
    return re.sub(r"\s+", " ", text).strip()


def _parse_json_list(raw: str) -> list[str] | None:
    """Find a JSON list of strings anywhere in the model output."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[", raw):
        try:
            data, _ = decoder.raw_decode(raw[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(isinstance(c, str) for c in data):
            return [c.strip() for c in data if c.strip()]
    return None

# if parse jason list fails to split llm output that handle by fallback to ensure pipeline runs.
def _fallback_split(text: str) -> list[str]:
    """If the LLM output is unusable, fall back to one claim per sentence."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]


def extract_claims(answer: str) -> list[str]:
    """Return a list of atomic claims found in the answer."""
    answer = clean_answer(answer)
    if any(p in answer.lower() for p in NO_INFO_PATTERNS) and len(answer.split()) < 40:
        return []

    prompt = f"{EXAMPLE}\n\nText: {answer}\nOutput:"
    raw = call_llm(prompt, system=SYSTEM, temperature=0.0, max_tokens=800)

    claims = _parse_json_list(raw)
    if claims is None:
        print("  (claim parsing failed, using sentence split instead)")
        print(f"  raw LLM output was: {raw[:300]!r}")
        claims = _fallback_split(answer)

    # remove duplicates, keep order
    seen, unique = set(), []
    for c in claims:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or (
        "The liver is the largest internal organ. It filters blood, produces bile, "
        "and stores glycogen. It is located in the upper right abdomen."
    )
    print(f"Answer: {text}\n")
    for i, claim in enumerate(extract_claims(text), 1):
        print(f"{i}. {claim}")

