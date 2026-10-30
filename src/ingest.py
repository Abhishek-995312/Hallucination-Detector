"""Phase 1a - Build the knowledge base.

Fetch Wikipedia articles -> clean -> chunk -> embed -> store in ChromaDB.

Run from the project root:
    python -m src.ingest
"""

import re

import chromadb
import requests
import time
from sentence_transformers import SentenceTransformer

from src import config

# Sections at the end of Wikipedia articles that contain no useful facts.
STOP_SECTIONS = r"\n==+\s*(See also|References|External links|Further reading|Notes|Bibliography|Sources|Citations)\s*==+"


def fetch_article(title: str) -> str:
    """Download plain-text article from Wikipedia (cached in data/raw)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.RAW_DIR / f"{title.replace('/', '_')}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }

    for attempt in range(5):
        resp = requests.get(
            config.WIKI_API, params=params,
            headers={"User-Agent": config.USER_AGENT}, timeout=30,
        )
        if resp.status_code == 429:  # too many requests: wait, then retry
            try:
                wait = int(resp.headers.get("Retry-After", ""))
            except ValueError:
                wait = 10 * (attempt + 1)
            print(f"  rate limited on '{title}', waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError("still rate limited after 5 tries")

    page = next(iter(resp.json()["query"]["pages"].values()))
    text = page.get("extract", "")
    cache.write_text(text, encoding="utf-8")
    time.sleep(1.5)  # be polite: pause between downloads
    return text


def clean_text(text: str) -> str:
    """Remove reference sections, headings markup and extra whitespace."""
    text = re.split(STOP_SECTIONS, text, maxsplit=1)[0]
    text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.MULTILINE)  # heading lines
    text = re.sub(r"\[\d+\]", "", text)  # [1] citations
    text = re.sub(r"\s*\n\s*", " ", text)  # newlines -> space
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, max_words: int, overlap_words: int, min_words: int) -> list[str]:
    """Sentence-aware chunking with overlap, so facts are not cut mid-sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        n = len(sent.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            # carry the last few sentences into the next chunk (overlap)
            keep, kept = [], 0
            for prev in reversed(current):
                pn = len(prev.split())
                if kept + pn > overlap_words:
                    break
                keep.insert(0, prev)
                kept += pn
            current, count = keep, kept
        current.append(sent)
        count += n

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= min_words]


def main() -> None:
    model = SentenceTransformer(config.EMBED_MODEL)

    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:  # rebuild from scratch so re-running never creates duplicates
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for title in config.TOPICS:
        try:
            raw = fetch_article(title)
        except Exception as e:
            print(f"[skip] {title}: {e}")
            continue

        chunks = chunk_text(
            clean_text(raw), config.CHUNK_WORDS,
            config.OVERLAP_WORDS, config.MIN_CHUNK_WORDS,
        )
        if not chunks:
            print(f"[skip] {title}: no usable text")
            continue

        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        collection.add(
            ids=[f"{title}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": title, "url": url, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
        total += len(chunks)
        print(f"[ok] {title}: {len(chunks)} chunks")

    print(f"\nDone. {total} chunks stored in {config.DB_DIR}")


if __name__ == "__main__":
    main()
