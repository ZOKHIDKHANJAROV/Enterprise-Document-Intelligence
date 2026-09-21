"""Extract text and metadata from enterprise documents.

Supported formats:
- PDF
- DOCX
- XLSX/XLS
- TXT/MD

The script recursively scans a directory and writes one JSON object per
document to a JSONL corpus. Binary documents and source files must stay outside
Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
from tqdm import tqdm


SUPPORTED = {".pdf", ".docx", ".xlsx", ".xlsm", ".txt", ".md"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[PAGE {index}]\n{text}")
    return clean_text("\n\n".join(pages)), len(reader.pages)


def extract_docx(path: Path) -> tuple[str, int]:
    document = Document(str(path))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        value = paragraph.text.strip()
        if value:
            blocks.append(value)

    for table_index, table in enumerate(document.tables, start=1):
        blocks.append(f"[TABLE {table_index}]")
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            blocks.append(" | ".join(cells))

    return clean_text("\n".join(blocks)), len(document.paragraphs)


def extract_xlsx(path: Path) -> tuple[str, int]:
    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    blocks: list[str] = []
    rows = 0

    try:
        for sheet in workbook.worksheets:
            blocks.append(f"[SHEET: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                if any(value.strip() for value in values):
                    blocks.append(" | ".join(values))
                    rows += 1
    finally:
        workbook.close()

    return clean_text("\n".join(blocks)), rows


def extract_text_file(path: Path) -> tuple[str, int]:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore")), 1


def extract(path: Path) -> tuple[str, int]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".docx":
        return extract_docx(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return extract_xlsx(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return extract_text_file(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def iter_documents(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/extracted/corpus.jsonl"))
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    documents = list(iter_documents(args.input_dir))
    if args.max_documents is not None:
        documents = documents[: args.max_documents]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "empty": 0,
        "failed": 0,
        "errors": [],
        "extensions": {},
    }

    with args.output.open("w", encoding="utf-8") as output:
        for path in tqdm(documents, desc="Extracting"):
            extension = path.suffix.lower()
            stats["extensions"][extension] = stats["extensions"].get(extension, 0) + 1

            try:
                text, units = extract(path)
                digest = sha256_file(path)

                record = {
                    "document_id": digest,
                    "file_name": path.name,
                    "relative_path": str(path.relative_to(args.input_dir)),
                    "extension": extension,
                    "text": text,
                    "text_length": len(text),
                    "units": units,
                    "sha256": digest,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                }

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["processed"] += 1

                if not text:
                    stats["empty"] += 1

            except Exception as exc:
                stats["failed"] += 1
                stats["errors"].append(
                    {"file": str(path), "error": repr(exc)}
                )

    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
