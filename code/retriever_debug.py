"""Interactive retriever smoke-test script."""

import argparse
from datetime import datetime
from pathlib import Path

from . import config
from .corpus_loader import CorpusLoader
from .models import Company
from .retriever import Retriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive retriever debugger for support-ticket queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--company",
        default="NONE",
        help="Company namespace to search: HACKERRANK, CLAUDE, VISA, or NONE.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of retrieved documents to print.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Similarity threshold passed to Retriever.retrieve().",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=config.PROJECT_ROOT / "logs" / "retriever_debug.log",
        help="Path to the log file where query results will be appended.",
    )
    return parser.parse_args()


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _append_log(
    log_file: Path,
    query: str,
    threshold_met: bool,
    docs: list,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")

    lines = [
        f"## [{ts}] query={query}",
        f"threshold_passed={threshold_met}",
    ]

    if not docs:
        lines.append("No documents retrieved.")
    else:
        for idx, retrieved in enumerate(docs, start=1):
            doc = retrieved.document
            product_area = doc.meta.get("product_area", "general")
            lines.extend(
                [
                    (
                        f"[{idx}] score={retrieved.score:.3f} "
                        f"company={doc.source.value} area={product_area}"
                    ),
                    f"id: {doc.id}",
                    f"preview: {_preview(doc.content)}",
                ]
            )

    with log_file.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n\n")


def main() -> None:
    args = _parse_args()
    company = Company.try_from_str(args.company)
    log_file = args.log_file

    print("Loading corpus and building retrieval index...")
    documents = CorpusLoader().load()
    retriever = Retriever(documents)

    print(
        f"Ready. company={company.value} top_k={args.top_k} threshold={args.threshold:.3f}"
    )
    print(f"Logging results to: {log_file}")
    print("Enter a query. Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nquery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            print("Please enter a non-empty query.")
            continue

        docs, threshold_met = retriever.retrieve(
            query=query,
            company=company,
            top_k=args.top_k,
            threshold=args.threshold,
        )

        print(f"threshold_passed={threshold_met}")
        if not docs:
            print("No documents retrieved.")
            _append_log(log_file, query, threshold_met, docs)
            continue

        for idx, retrieved in enumerate(docs, start=1):
            doc = retrieved.document
            product_area = doc.meta.get("product_area", "general")
            print(
                f"\n[{idx}] score={retrieved.score:.3f} "
                f"company={doc.source.value} area={product_area}"
            )
            print(f"id: {doc.id}")
            print(f"preview: {_preview(doc.content)}")

        _append_log(log_file, query, threshold_met, docs)


if __name__ == "__main__":
    main()
