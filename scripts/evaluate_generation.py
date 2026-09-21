import argparse
import json
import re
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()

    rows, scores, exact = [], [], 0.0

    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score = overlap(row["prediction"], row["reference"])
            match = normalize(row["prediction"]) == normalize(row["reference"])
            scores.append(score)
            exact += float(match)
            rows.append({**row, "exact_match": match, "token_recall": score})

    count = len(rows)
    summary = {
        "examples": count,
        "exact_match": exact / count if count else 0.0,
        "mean_token_recall": sum(scores) / count if count else 0.0,
        "warning": "Lexical metrics are a baseline only.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "examples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
