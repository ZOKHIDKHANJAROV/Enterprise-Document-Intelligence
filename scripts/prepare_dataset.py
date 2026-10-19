"""Validate and normalize an instruction-tuning JSONL dataset.

Expected input records:
{"document_id": "...", "document_type": "...", "instruction": "...",
 "input": "...", "output": "..."}

document_id is required because the train/validation/test split is done by
document. document_type defaults to "unknown".

The script does not generate training examples. It validates the dataset so
QLoRA is not started against malformed or empty records. With --max-noise it
also drops pairs whose context or answer is mostly OCR noise.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from text_quality import noise_ratio


REQUIRED_FIELDS = ("document_id", "instruction", "output")


def join_list_answer(output: str) -> str:
    """Models sometimes return a list; it was saved as "['a', 'b']"."""
    text = output.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return output
        if isinstance(items, list) and all(isinstance(item, str) for item in items):
            return "; ".join(item.strip() for item in items if item.strip())
    return output


def is_noisy(record: dict, max_noise: float) -> bool:
    return noise_ratio(record["input"]) > max_noise or noise_ratio(record["output"]) > max_noise


def normalize_record(record: dict, line_number: int) -> dict:
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if field == "document_id" and isinstance(value, int):
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number}: '{field}' must be a non-empty string."
            )

    input_text = record.get("input", "")
    if input_text is None:
        input_text = ""
    if not isinstance(input_text, str):
        raise ValueError(f"Line {line_number}: 'input' must be a string.")

    document_type = record.get("document_type") or "unknown"
    if not isinstance(document_type, str):
        raise ValueError(f"Line {line_number}: 'document_type' must be a string.")

    normalized = {
        "document_id": str(record["document_id"]).strip(),
        "document_type": document_type.strip(),
        "instruction": record["instruction"].strip(),
        "input": input_text.strip(),
        "output": join_list_answer(record["output"]).strip(),
    }
    # Keep the generation grounding score for dataset_report.py.
    if isinstance(record.get("grounding"), (int, float)):
        normalized["grounding"] = record["grounding"]
    return normalized


def validate_jsonl(input_path: Path, output_path: Path, max_noise: float | None = None,
                   stats: dict | None = None) -> int:
    count = 0
    stats = stats if stats is not None else {}
    stats.setdefault("noisy", 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")

            normalized = normalize_record(record, line_number)
            if max_noise is not None and is_noisy(normalized, max_noise):
                stats["noisy"] += 1
                continue
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-noise",
        type=float,
        default=None,
        help="Drop pairs whose context or answer has a larger share of "
        "OCR-noise tokens (0.3 is a good start).",
    )
    args = parser.parse_args()

    stats: dict = {}
    count = validate_jsonl(args.input, args.output, args.max_noise, stats)
    print(f"Validated records: {count}")
    print(f"Dropped as OCR noise: {stats['noisy']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
