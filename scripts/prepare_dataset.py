"""Validate and normalize an instruction-tuning JSONL dataset.

Expected input records:
{"instruction": "...", "input": "...", "output": "..."}

The script does not generate training examples. It validates the dataset so
QLoRA is not started against malformed or empty records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = ("instruction", "output")


def normalize_record(record: dict, line_number: int) -> dict:
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number}: '{field}' must be a non-empty string."
            )

    input_text = record.get("input", "")
    if input_text is None:
        input_text = ""
    if not isinstance(input_text, str):
        raise ValueError(f"Line {line_number}: 'input' must be a string.")

    return {
        "instruction": record["instruction"].strip(),
        "input": input_text.strip(),
        "output": record["output"].strip(),
    }


def validate_jsonl(input_path: Path, output_path: Path) -> int:
    count = 0
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
            dst.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    count = validate_jsonl(args.input, args.output)
    print(f"Validated records: {count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
