#!/usr/bin/env python3
"""
Query the Math & Quant AI knowledge base from the terminal.

Usage:
    python scripts/query.py "Explain the Sharpe Ratio"
    python scripts/query.py "Derive the Black-Scholes equation" --top-k 8
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import query


def main():
    parser = argparse.ArgumentParser(description="Query the Math & Quant AI")
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    args = parser.parse_args()

    user_query = " ".join(args.question)

    print(f"\nQuery: {user_query}")
    print("─" * 60)

    result = query(user_query, top_k=args.top_k)

    print(f"\nAnswer:\n{result['answer']}")

    print(f"\nSources used ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  • {s['source']}  —  page {s['page_num']}  (score: {s['score']})")


if __name__ == "__main__":
    main()
