"""Crawl a folder and refresh the index."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from local_rag.chunking import chunk_text, metadata_card, normalize
from local_rag.extract import extract, kind_for
from local_rag.settings import MODEL_ID
from local_rag import store

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"}
# Bump when extraction changes, so unchanged files are read again.
EXTRACTOR_REV = "3"
MAX_FILE_BYTES = 80 * 1024 * 1024
MAX_CHUNKS = 40


def crawl(root: Path, max_files: int = 400) -> list[Path]:
    root = root.resolve()
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".") and name not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if kind_for(path) is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or size == 0:
                continue
            found.append(path)
            if len(found) >= max_files:
                return found
    return found


def fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}:{MODEL_ID}:{EXTRACTOR_REV}"


def index_folder(root: Path | str, max_files: int = 400, on_event=None) -> dict:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    if root == Path("/") or root == Path.home():
        raise ValueError("Refusing to index / or your whole home folder. Pick a smaller folder.")

    def emit(message: str):
        if on_event:
            on_event(message)

    files = crawl(root, max_files=max_files)
    emit(f"Found {len(files)} supported files under {root}")
    if len(files) >= max_files:
        emit(f"Stopped at {max_files} files. Raise the limit if this folder is larger.")

    kept = {str(path) for path in files}
    removed = store.delete_missing(kept)
    if removed:
        emit(f"Removed {removed} files that are no longer in the folder")

    indexed = 0
    skipped = 0
    failed = 0
    for path in files:
        fp = fingerprint(path)
        if store.indexed_fingerprint(str(path)) == fp:
            skipped += 1
            continue
        emit(f"Reading {path.name}")
        try:
            _index_one(path, fp)
            indexed += 1
        except Exception as exc:
            failed += 1
            emit(f"Failed {path.name}: {exc}")
    emit(f"Done. Indexed {indexed}, unchanged {skipped}, failed {failed}.")
    return {"found": len(files), "indexed": indexed, "unchanged": skipped, "failed": failed, "root": str(root)}


def _index_one(path: Path, fp: str) -> None:
    extracted = extract(path)
    pieces = []
    card = metadata_card(
        path.name,
        extracted.kind,
        extracted.captured_at,
        extracted.time_source,
        extracted.country,
        extracted.lat,
        extracted.lon,
    )
    pieces.append(card)
    body = normalize(extracted.text)
    content_chunks = chunk_text(body) if body else []
    truncated = False
    if len(content_chunks) > MAX_CHUNKS - 1:
        content_chunks = content_chunks[: MAX_CHUNKS - 1]
        truncated = True
    pieces.extend(content_chunks)
    from local_rag.embedder import embed_texts

    vectors = embed_texts(pieces)
    chunk_rows = []
    for ordinal, (text, vector) in enumerate(zip(pieces, vectors)):
        chunk_rows.append(
            {
                "path": str(path),
                "ordinal": ordinal,
                "text": text,
                "kind": extracted.kind,
                "captured_at": extracted.captured_at,
                "country": extracted.country,
                "vector": np.asarray(vector, dtype=np.float32).tobytes(),
            }
        )
    warnings = list(extracted.warnings)
    if truncated:
        warnings.append(f"kept the first {MAX_CHUNKS} chunks")
    preview_source = body or card
    store.replace_file(
        {
            "path": str(path),
            "name": path.name,
            "kind": extracted.kind,
            "fingerprint": fp,
            "captured_at": extracted.captured_at,
            "time_source": extracted.time_source,
            "country": extracted.country,
            "lat": extracted.lat,
            "lon": extracted.lon,
            "chars": len(body),
            "ocr_used": 1 if extracted.ocr_used else 0,
            "warning": "; ".join(warnings),
            "preview": preview_source[:280],
        },
        chunk_rows,
    )
