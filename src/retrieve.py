"""Phase 1b - Retrieve evidence for a query.

Usage as a library (used later by the pipeline):
    from src.retrieve import Retriever
    r = Retriever()
    hits = r.search("What does the liver do?", k=5)

Quick test from the project root:
    python -m src.retrieve "What does the liver do?"
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from src import config


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        self.collection = client.get_collection(config.COLLECTION_NAME)

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[dict]:
        """Return top-k chunks as dicts: text, source, url, chunk_index, score.

        score = cosine similarity (higher is more relevant).
        """
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=k)

        hits = []
        for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = 1.0 - dist
            if score >= min_score:
                hits.append({
                    "text": doc,
                    "source": meta["source"],
                    "url": meta["url"],
                    "chunk_index": meta["chunk_index"],
                    "score": round(score, 4),
                })
        return hits


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What does the liver do?"
    retriever = Retriever()
    print(f"Query: {query}\n")
    for i, h in enumerate(retriever.search(query, k=5), 1):
        print(f"[{i}] {h['source']} (chunk {h['chunk_index']}, score {h['score']})")
        print(f"    {h['text'][:300]}...\n")

