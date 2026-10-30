"""Shared settings for Phase 1 (knowledge base + retrieval)."""

import os
from dotenv import load_dotenv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT /".env")

RAW_DIR = ROOT / "data" / "raw"  # downloaded Wikipedia text (cache)
DB_DIR = ROOT / "data" / "chroma_db"  # ChromaDB persistent storage
COLLECTION_NAME = "human_body"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_WORDS = 200  # target max words per chunk
OVERLAP_WORDS = 40  # words shared between neighbouring chunks
MIN_CHUNK_WORDS = 30  # drop tiny leftovers

def _require_env(name:str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env file (not set)")
    return value

WIKI_API = _require_env("WIKI_API")
USER_AGENT = _require_env("WIKI_USER_AGENT")

# Wikipedia articles about the human body. Edit freely.
TOPICS = [
    "Human body", "Human brain", "Heart", "Lung", "Liver", "Kidney",
    "Stomach", "Small intestine", "Large intestine", "Pancreas",
    "Spleen", "Human skin", "Human skeleton", "Skeletal muscle",
    "Blood", "Red blood cell", "White blood cell", "Platelet",
    "Blood pressure", "Circulatory system", "Respiratory system",
    "Digestive system", "Nervous system", "Immune system",
    "Endocrine system", "Lymphatic system", "Urinary system",
    "Reproductive system", "Human eye", "Ear", "Spinal cord",
    "Thyroid", "Insulin", "Hemoglobin", "Bone", "Joint",
    "Human digestive system", "Human heart", "Metabolism",
]
