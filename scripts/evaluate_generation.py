import argparse
import json
import re
from pathlib import Path

from add_unanswerable import is_refusal


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\sа-яё]", "", text, flags=re.IGNORECASE)


def token_set(text):
    return set(normalize(text).split())


def overlap(prediction, reference):
    predicted = token_set(prediction)
    expected = token_set(reference)
    return len(predicted & expected) / len(expected) if expected else 0.0


def rate(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    answerable = [r for r in rows if r.get("answerable", True)]
    unanswerable = [r for r in rows if not r.get("answerable", True)]
    return {
        "examples": len(rows),
        "answerable_examples": len(answerable),
        "unanswerable_examples": len(unanswerable),
        # Lexical metrics are computed on answerable questions only.
        "exact_match": rate([r["exact_match"] for r in answerable]),
        "mean_token_recall": rate([r["token_recall"] for r in answerable]),
        # Share of unanswerable questions where the model correctly refused.
        "refusal_accuracy": rate([r["refusal"] for r in unanswerable]),
        # Share of answerable questions where the model wrongly refused.
        "false_refusal_rate": rate([r["refusal"] for r in answerable]),
        "warning": "Lexical and refusal metrics are a baseline only; see judge_predictions.py.",
    }


def score_row(row):
    match = normalize(row["prediction"]) == normalize(row["reference"])
    return {**row, "exact_match": match, "token_recall": overlap(row["prediction"], row["reference"]),
            "refusal": is_refusal(row["prediction"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()

    rows = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(score_row(json.loads(line)))

    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "examples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
