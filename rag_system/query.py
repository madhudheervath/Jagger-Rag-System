"""
query.py — CLI for interactive single-question querying.

Usage:
    python query.py "What is the authority level under Navy Regulations Part IV?"
    python query.py  # (interactive prompt mode)
"""

import sys
import textwrap

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag_core
import config


def print_result(result: dict) -> None:
    """Pretty-print a RAG result to the terminal."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {result['question']}")
    print("=" * 70)

    answer = result["answer"]
    if answer == config.CANNOT_ANSWER_SENTINEL:
        print("\n[UNANSWERABLE] The corpus does not contain sufficient information.")
    else:
        print("\nANSWER:")
        for line in textwrap.wrap(answer, width=68):
            print(f"   {line}")

    print(f"\nSOURCE  : {result['source'] or 'N/A'}")
    print(f"SECTION : {result['section'] or 'N/A'}")

    print(f"\nTOP RETRIEVED CHUNKS (k={len(result['chunks'])}):")
    for i, (chunk, score) in enumerate(zip(result["chunks"], result["scores"]), 1):
        doc = chunk.get("document", "?")
        sec = chunk.get("section", "?")
        text_preview = chunk.get("text", "")[:120].replace("\n", " ")
        print(f"  [{i}] score={score:.4f} | {doc} | {sec}")
        print(f"       {text_preview}...")

    print("=" * 70 + "\n")


def main() -> None:
    # Preload index
    print("Loading index...", end=" ", flush=True)
    try:
        rag_core.load_index()
        print("done.")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI arg or interactive mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = rag_core.answer_question(question)
        print_result(result)
    else:
        print(f"\nDefence RAG -- querying: {config.TARGET_DOCUMENT}")
        print("Type a question (or 'quit' to exit).\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            result = rag_core.answer_question(question)
            print_result(result)


if __name__ == "__main__":
    main()
