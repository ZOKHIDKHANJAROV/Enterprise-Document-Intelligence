"""Inspect an extracted JSONL corpus before dataset generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-text-length", type=int, default=100)
    args = parser.parse_args()

    total = 0
    empty = 0
    short = 0
    methods = Counter()
    extensions = Counter()
    lengths: list[int] = []
    suspicious: list[dict] = []

    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            total += 1

            text = str(record.get("text", ""))
            length = len(text)
            lengths.append(length)

            extension = record.get("extension", "unknown")
            method = record.get("extraction_method", "unknown")
            extensions[extension] += 1
            methods[method] += 1

            if not text.strip():
                empty += 1

            if length < args.min_text_length:
                short += 1
                if len(suspicious) < args.top:
                    suspicious.append(
                        {
                            "line": line_number,
                            "document_id": record.get("document_id"),
                            "file_name": record.get("file_name"),
                            "extension": extension,
                            "length": length,
                            "method": method,
                        }
                    )

    lengths_sorted = sorted(lengths)

    def percentile(values: list[int], p: float) -> int:
        if not values:
            return 0
        index = min(
            len(values) - 1,
            max(0, int(round((len(values) - 1) * p))),
        )
        return values[index]

    report = {
        "total_documents": total,
        "empty_documents": empty,
        "short_documents": short,
        "short_threshold": args.min_text_length,
        "empty_rate": round(empty / total, 4) if total else 0,
        "short_rate": round(short / total, 4) if total else 0,
        "extraction_methods": dict(methods),
        "extensions": dict(extensions),
        "text_length": {
            "min": min(lengths) if lengths else 0,
            "p50": percentile(lengths_sorted, 0.50),
            "p90": percentile(lengths_sorted, 0.90),
            "p99": percentile(lengths_sorted, 0.99),
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
        "suspicious_examples": suspicious,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
