"""
rag_core.py — Core retrieval and generation logic for the Defence RAG system.

Uses the new google.genai SDK (replaces deprecated google.generativeai).

Provides:
  - load_index()          : load FAISS index + metadata from disk
  - embed_query(text)     : embed a question with RETRIEVAL_QUERY task type
  - mmr_rerank(...)       : Maximal Marginal Relevance re-ranking
  - retrieve(question)    : FAISS search → MMR → top-k chunks
  - generate_answer(...)  : call Gemini flash with structured JSON prompt
  - answer_question(...)  : end-to-end pipeline returning a result dict
"""

import json
import pickle
import re
import time
from typing import Any

import faiss
import numpy as np
from google import genai
from google.genai import types

import config

# ── Gemini client ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=config.GEMINI_API_KEY)

# ── Index cache (loaded once per process) ─────────────────────────────────────
_index: Any = None
_metadata: list[dict] | None = None


def load_index() -> tuple[Any, list[dict]]:
    """Load FAISS index and metadata from disk. Cached after first call."""
    global _index, _metadata
    if _index is not None and _metadata is not None:
        return _index, _metadata

    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {config.FAISS_INDEX_PATH}. "
            "Run `python ingest.py` first."
        )

    _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PKL_PATH, "rb") as f:
        _metadata = pickle.load(f)

    return _index, _metadata


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string with RETRIEVAL_QUERY task type.
    Returns a (1, 3072) float32 array, L2-normalised.
    """
    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise


# ── MMR re-ranking ────────────────────────────────────────────────────────────

def mmr_rerank(
    candidate_indices: list[int],
    candidate_scores: list[float],
    all_embeddings: np.ndarray,
    k: int = 8,
    lam: float = 0.6,
) -> list[int]:
    """
    Maximal Marginal Relevance re-ranking.
    Selects k indices balancing relevance vs. diversity.
    lam=1.0 → pure relevance; lam=0.0 → pure diversity.
    """
    if len(candidate_indices) <= k:
        return candidate_indices

    selected: list[int] = []
    remaining = list(zip(candidate_indices, candidate_scores))

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -np.inf

        for idx, rel_score in remaining:
            if not selected:
                mmr_score = lam * rel_score
            else:
                selected_vecs = all_embeddings[selected]
                cand_vec = all_embeddings[idx].reshape(1, -1)
                sim_to_selected = float(np.max(cand_vec @ selected_vecs.T))
                mmr_score = lam * rel_score - (1 - lam) * sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining = [(i, s) for i, s in remaining if i != best_idx]

    return selected


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str) -> list[dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a question.
    1. Embed question (RETRIEVAL_QUERY)
    2. FAISS search → MMR_CANDIDATES candidates
    3. MMR re-rank → TOP_K diverse chunks
    """
    index, metadata = load_index()

    q_vec = embed_query(question)
    n_candidates = min(config.MMR_CANDIDATES, index.ntotal)

    scores, indices = index.search(q_vec, n_candidates)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    valid = [(i, s) for i, s in zip(indices, scores) if i >= 0]
    if not valid:
        return []

    candidate_indices, candidate_scores = zip(*valid)

    # Reconstruct embeddings for MMR
    all_vecs = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for j in range(index.ntotal):
        all_vecs[j] = index.reconstruct(j)

    mmr_indices = mmr_rerank(
        candidate_indices=list(candidate_indices),
        candidate_scores=list(candidate_scores),
        all_embeddings=all_vecs,
        k=config.TOP_K,
        lam=config.MMR_LAMBDA,
    )

    score_map = dict(zip(candidate_indices, candidate_scores))
    results = []
    for idx in mmr_indices:
        chunk = dict(metadata[idx])
        chunk["_score"] = score_map.get(idx, 0.0)
        results.append(chunk)

    return results


# ── Generation ────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        doc = chunk.get("document", "Unknown")
        section = chunk.get("section", "Unknown Section")
        text = chunk.get("text", "").strip()
        lines.append(f"[{i}] Source: {doc} | Section: {section}\n{text}")
    return "\n\n---\n\n".join(lines)


def generate_answer(question: str, chunks: list[dict]) -> dict[str, str]:
    """
    Generate a grounded answer using Gemini flash.
    Returns dict: answer, source, section, raw_response.
    """
    if not chunks:
        return {
            "answer": config.CANNOT_ANSWER_SENTINEL,
            "source": "",
            "section": "",
            "raw_response": "",
        }

    context_block = _build_context_block(chunks)

    prompt = f"""{config.SYSTEM_PROMPT}

=== CONTEXT ===
{context_block}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Respond ONLY with a valid JSON object. No markdown, no code fences, no extra text.
Format exactly:
{{"answer": "<your concise answer>", "source": "<document filename>", "section": "<section heading>"}}

If you cannot answer from the context, use:
{{"answer": "{config.CANNOT_ANSWER_SENTINEL}", "source": "", "section": ""}}"""

    raw = ""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=config.GEN_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            break
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return {
                    "answer": config.CANNOT_ANSWER_SENTINEL,
                    "source": "",
                    "section": "",
                    "raw_response": str(e),
                }

    # ── Parse JSON ────────────────────────────────────────────────────────────
    parsed = None
    try:
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean, flags=re.IGNORECASE).strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?"answer"\s*:.*?\}', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed and isinstance(parsed, dict):
        answer = str(parsed.get("answer", "")).strip()
        source = str(parsed.get("source", "")).strip()
        section = str(parsed.get("section", "")).strip()
    else:
        answer = raw if raw else config.CANNOT_ANSWER_SENTINEL
        source = chunks[0].get("document", "") if chunks else ""
        section = chunks[0].get("section", "") if chunks else ""

    # Ensure source points to our known document if it was cited
    if source and config.TARGET_DOCUMENT not in source and chunks:
        source = chunks[0].get("document", "")

    return {
        "answer": answer,
        "source": source,
        "section": section,
        "raw_response": raw,
    }


# ── End-to-end ────────────────────────────────────────────────────────────────

def answer_question(question: str) -> dict[str, Any]:
    """Full RAG pipeline for a single question."""
    chunks = retrieve(question)
    result = generate_answer(question, chunks)

    return {
        "question": question,
        "answer": result["answer"],
        "source": result["source"],
        "section": result["section"],
        "chunks": chunks,
        "scores": [c.get("_score", 0.0) for c in chunks],
        "raw_response": result.get("raw_response", ""),
    }
