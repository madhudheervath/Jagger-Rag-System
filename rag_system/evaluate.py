"""
evaluate.py — Batch evaluation pipeline for the Kaggle competition.

What it does:
1. Reads test.csv (140 questions)
2. Runs each question through the full RAG pipeline
3. Writes submission.csv in competition format (id, prediction, pred_source, pred_section)
4. Computes Context Hit Rate — our retrieval quality metric

Usage:
    python evaluate.py
    python evaluate.py --limit 10          # test on first 10 questions only
    python evaluate.py --output my_sub.csv # custom output path
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from tqdm import tqdm

import rag_core
import config


# ── Metric: Context Hit Rate ───────────────────────────────────────────────────

def compute_context_hit_rate(
    retrieved_chunks_list: list[list[dict]],
    gold_contexts_list: list[str],
) -> float:
    """
    Context Hit Rate: for each question, did at least one retrieved chunk
    contain a substring from any of the gold context sentences?

    The test.csv `contexts` column contains gold sentences separated by |||.

    Returns a float in [0, 1].
    """
    hits = 0
    total = 0

    for chunks, gold_str in zip(retrieved_chunks_list, gold_contexts_list):
        if not isinstance(gold_str, str) or not gold_str.strip():
            continue  # skip questions without gold contexts

        gold_sentences = [s.strip().lower() for s in gold_str.split("|||") if s.strip()]
        if not gold_sentences:
            continue

        total += 1
        retrieved_texts = " ".join(
            c.get("text", "").lower() for c in chunks
        )

        # Hit = any gold sentence appears (as substring) in retrieved text
        if any(gs[:60] in retrieved_texts for gs in gold_sentences):
            hits += 1

    return hits / total if total > 0 else 0.0


# ── Source-match accuracy ──────────────────────────────────────────────────────

def compute_source_accuracy(
    pred_sources: list[str],
    questions: list[str],
    target_doc: str,
) -> dict:
    """
    Count how many questions that mention the target document received
    the correct pred_source.
    """
    correct = 0
    relevant = 0
    for q, ps in zip(questions, pred_sources):
        doc_name_keyword = target_doc.replace(".pdf", "").lower()
        if doc_name_keyword in q.lower():
            relevant += 1
            if target_doc in ps:
                correct += 1
    return {"relevant_questions": relevant, "correct_source": correct}


# ── Main evaluation loop ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system on test.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--output", type=str, default=str(config.SUBMISSION_CSV), help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between API calls")
    args = parser.parse_args()

    # ── Load index ─────────────────────────────────────────────────────────────
    print("Loading FAISS index...", end=" ", flush=True)
    try:
        rag_core.load_index()
        print("done.")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Load test questions ────────────────────────────────────────────────────
    print(f"Reading {config.TEST_CSV}...")
    test_df = pd.read_csv(config.TEST_CSV)
    if args.limit:
        test_df = test_df.head(args.limit)
    print(f"  {len(test_df)} questions to process.")

    # ── Run RAG pipeline ───────────────────────────────────────────────────────
    predictions = []
    pred_sources = []
    pred_sections = []
    all_retrieved_chunks = []
    all_gold_contexts = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Answering questions"):
        question = str(row["question"])
        gold_contexts = str(row.get("contexts", ""))

        result = rag_core.answer_question(question)

        predictions.append(result["answer"])
        pred_sources.append(result["source"])
        pred_sections.append(result["section"])
        all_retrieved_chunks.append(result["chunks"])
        all_gold_contexts.append(gold_contexts)

        # Polite delay to avoid rate limiting
        time.sleep(args.delay)

    # ── Write submission.csv ───────────────────────────────────────────────────
    submission_df = pd.DataFrame({
        "id": test_df["id"].values,
        "prediction": predictions,
        "pred_source": pred_sources,
        "pred_section": pred_sections,
    })
    output_path = Path(args.output)
    submission_df.to_csv(output_path, index=False)
    print(f"\n[DONE] Submission saved -> {output_path}")
    print(f"   Rows: {len(submission_df)}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    # 1. Context Hit Rate
    hit_rate = compute_context_hit_rate(all_retrieved_chunks, all_gold_contexts)
    print(f"\n1. Context Hit Rate (retrieval quality): {hit_rate:.1%}")
    print(f"   (proportion of questions where ≥1 retrieved chunk")
    print(f"    contains a gold context sentence substring)")

    # 2. Source Accuracy
    src_acc = compute_source_accuracy(
        pred_sources, list(test_df["question"]), config.TARGET_DOCUMENT
    )
    if src_acc["relevant_questions"] > 0:
        acc = src_acc["correct_source"] / src_acc["relevant_questions"]
        print(f"\n2. Source Accuracy (on {config.TARGET_DOCUMENT} questions):")
        print(f"   {src_acc['correct_source']}/{src_acc['relevant_questions']} = {acc:.1%}")
    else:
        print(f"\n2. Source Accuracy: No questions explicitly mention {config.TARGET_DOCUMENT}")

    # 3. Unanswerable rate
    cannot_count = sum(
        1 for p in predictions
        if config.CANNOT_ANSWER_SENTINEL in p or p.strip() == ""
    )
    print(f"\n3. Unanswerable rate: {cannot_count}/{len(predictions)} = {cannot_count/len(predictions):.1%}")
    print(f"   (questions where corpus had insufficient info)")

    # 4. Answer non-empty rate
    non_empty = sum(1 for p in predictions if p.strip() and config.CANNOT_ANSWER_SENTINEL not in p)
    print(f"\n4. Answered rate: {non_empty}/{len(predictions)} = {non_empty/len(predictions):.1%}")

    print("\n" + "=" * 60)
    print("Sample predictions:")
    print("=" * 60)
    for i, row in submission_df.head(5).iterrows():
        print(f"\n  [{row['id']}] {test_df.iloc[i]['question'][:80]}...")
        print(f"       Answer: {str(row['prediction'])[:100]}...")
        print(f"       Source: {row['pred_source']} | Section: {row['pred_section']}")


if __name__ == "__main__":
    main()
