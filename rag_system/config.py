"""
config.py — Central configuration for the Defence RAG system.
All tuneable constants live here so changes are easy to make in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from this file's directory ──────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Models (new google.genai SDK) ────────────────────────────────────────────
EMBED_MODEL: str = "gemini-embedding-2"        # best available model for this API key
GEN_MODEL: str = "gemini-2.5-flash"            # switched from flash-lite (daily quota exhausted)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # Jagger/
DATA_DIR = BASE_DIR / "defence-rag-procurement-policy-reasoning-challenge"
METADATA_CSV = DATA_DIR / "metaData.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
SUBMISSION_CSV = BASE_DIR / "submission.csv"

INDEX_DIR = BASE_DIR / "rag_index"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PKL_PATH = INDEX_DIR / "metadata.pkl"

# ── Document filter ───────────────────────────────────────────────────────────
TARGET_DOCUMENT: str = "RegsNavyIV.pdf"

# ── Chunking / Retrieval ──────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = 50       # texts per embedding API call (conservative for new SDK)
TOP_K: int = 8                   # final chunks passed to LLM
MMR_CANDIDATES: int = 20         # initial FAISS candidates before MMR re-rank
MMR_LAMBDA: float = 0.6          # relevance weight in MMR (1.0 = pure similarity)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert on Indian Navy Regulations. \
Answer questions ONLY using the provided context chunks. \
Each chunk is tagged with its source document and section heading. \
If the context does not contain enough information to answer the question, \
respond with exactly: "The corpus does not contain sufficient information to answer this question." \
Do NOT invent facts or draw on outside knowledge. \
Be concise (2-3 sentences max) and always cite the source document and section."""

CANNOT_ANSWER_SENTINEL = (
    "The corpus does not contain sufficient information to answer this question."
)
