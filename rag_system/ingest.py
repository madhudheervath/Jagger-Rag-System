"""
ingest.py — Index RegsNavyIV.pdf chunks from metaData.csv into a FAISS vector store.

Usage:
    python ingest.py

What it does:
1. Reads metaData.csv and filters to TARGET_DOCUMENT (RegsNavyIV.pdf).
2. Embeds all chunk texts using gemini text-embedding-004 (RETRIEVAL_DOCUMENT task).
3. L2-normalises the vectors so FAISS IndexFlatIP == cosine similarity.
4. Saves the index and metadata to rag_index/ (idempotent — skips if already done).
"""

import pickle
import sys
import time

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from google import genai
from google.genai import types

import config

# ── Gemini client (new SDK) ────────────────────────────────────────────────────
client = genai.Client(api_key=config.GEMINI_API_KEY)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Embed a list of texts one-by-one using Gemini embedding model.
    Retry is per-text so rate-limit errors never cause duplicate embeddings.
    Returns an array of shape (len(texts), 3072), L2-normalised.
    """
    all_embeddings = []

    for text in tqdm(texts, desc="Embedding chunks"):
        for attempt in range(6):
            try:
                result = client.models.embed_content(
                    model=config.EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                all_embeddings.append(result.embeddings[0].values)
                time.sleep(0.6)   # stay within free-tier rate limit (~100 RPM)
                break
            except Exception as e:
                if attempt < 5:
                    wait = 2 ** (attempt + 1)
                    print(f"\n  Rate limit hit, waiting {wait}s before retry...", flush=True)
                    time.sleep(wait)
                else:
                    raise

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    arr = arr / norms
    return arr


def main() -> None:
    # ── Check if index already exists ─────────────────────────────────────────
    if config.FAISS_INDEX_PATH.exists() and config.METADATA_PKL_PATH.exists():
        print(f"Index already exists at {config.INDEX_DIR}. Skipping ingest.")
        print("Delete rag_index/ and re-run to rebuild.")
        return

    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Add it to rag_system/.env", file=sys.stderr)
        sys.exit(1)

    # ── Load and filter metaData.csv ──────────────────────────────────────────
    print(f"Loading {config.METADATA_CSV}...")
    df = pd.read_csv(config.METADATA_CSV, on_bad_lines="skip")
    print(f"  Total rows: {len(df):,}")

    df_filtered = df[df["document"] == config.TARGET_DOCUMENT].reset_index(drop=True)
    print(f"  Rows for '{config.TARGET_DOCUMENT}': {len(df_filtered):,}")

    if len(df_filtered) == 0:
        print(f"ERROR: No rows found for document '{config.TARGET_DOCUMENT}'.", file=sys.stderr)
        sys.exit(1)

    # ── Prepare texts for embedding ───────────────────────────────────────────
    texts = []
    for _, row in df_filtered.iterrows():
        section = str(row.get("section", "")).strip()
        topic = str(row.get("topic", "")).strip()
        text = str(row.get("text", "")).strip()
        combined = f"Section: {section}\nTopic: {topic}\n\n{text}" if section else text
        texts.append(combined)

    print(f"\nEmbedding {len(texts):,} chunks with '{config.EMBED_MODEL}'...")
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index: {index.ntotal:,} vectors (dim={dim})")

    # ── Save index + metadata ─────────────────────────────────────────────────
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"  FAISS index saved -> {config.FAISS_INDEX_PATH}")

    # Save the filtered dataframe as metadata (preserves document, section, text, chunk_id)
    metadata = df_filtered.to_dict(orient="records")
    with open(config.METADATA_PKL_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved  -> {config.METADATA_PKL_PATH}")

    print("\nIngest complete. Run: python query.py \"<your question>\"  to test.")


if __name__ == "__main__":
    main()
