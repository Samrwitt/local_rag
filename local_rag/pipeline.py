"""Index selected files now; folder crawl is ready for later auto-scale."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from local_rag.chunking import chunk_text, metadata_card, normalize
from local_rag.extract import extract, kind_for
from local_rag.settings import MODEL_ID
from local_rag import store

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache"}
EXTRACTOR_REV = "10"
MAX_FILE_BYTES = 80 * 1024 * 1024
MAX_CHUNKS = 40


def crawl(root: Path, max_files: int = 400) -> list[Path]:
    """Future path: auto-crawl every supported file under an authorized root."""
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


def content_hash(path: Path, sample_bytes: int = 2_000_000) -> str:
    """SHA-256 over head+tail samples — detects duplicates without full read of huge files."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        head = handle.read(sample_bytes)
        digest.update(head)
        if size > sample_bytes * 2:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
        elif size > sample_bytes:
            digest.update(handle.read())
    return digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Incremental skip key: size + mtime + model + extractor + content sample hash."""
    stat = path.stat()
    try:
        ch = content_hash(path)
    except OSError:
        ch = "unreadable"
    return f"{stat.st_size}:{stat.st_mtime_ns}:{MODEL_ID}:{EXTRACTOR_REV}:{ch[:16]}"


def index_paths(paths: list[Path | str], on_event=None) -> dict:
    """Index an explicit selection of files. Additive: other indexed files stay."""

    def emit(message: str):
        if on_event:
            on_event(message)

    resolved = []
    for raw in paths:
        # Keep the selected path as given so names like Italy_2019_rome.mp4
        # survive even when they are symlinks to a generic file.
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).absolute()
        if path.is_dir():
            emit(f"Skipping directory in file list: {path}")
            continue
        if not path.exists():
            emit(f"Missing: {path}")
            continue
        if kind_for(path) is None:
            emit(f"Unsupported type: {path.name}")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            emit(f"Unreadable {path.name}: {exc}")
            continue
        if size == 0 or size > MAX_FILE_BYTES:
            emit(f"Skipped by size: {path.name}")
            continue
        resolved.append(path)

    emit(f"Selected {len(resolved)} files to index")
    indexed = skipped = failed = 0
    for path in resolved:
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
    return {
        "mode": "selected",
        "found": len(resolved),
        "indexed": indexed,
        "unchanged": skipped,
        "failed": failed,
    }


def index_folder(root: Path | str, max_files: int = 400, on_event=None, purge_missing: bool = False) -> dict:
    """Crawl a folder. Access is treated as allowed for the running user.

    Later this becomes the auto-crawl of all authorized files. Today it is
    the batch path next to selected-file indexing.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    def emit(message: str):
        if on_event:
            on_event(message)

    files = crawl(root, max_files=max_files)
    emit(f"Found {len(files)} supported files under {root}")
    if len(files) >= max_files:
        emit(f"Stopped at {max_files} files. Raise the limit to crawl more.")

    if purge_missing:
        kept = {str(path) for path in files}
        removed = 0
        with store.connect() as conn:
            rows = conn.execute("SELECT path FROM files").fetchall()
            for row in rows:
                path = row["path"]
                if path.startswith(str(root) + os.sep) and path not in kept:
                    conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
                    conn.execute("DELETE FROM files WHERE path = ?", (path,))
                    removed += 1
        if removed:
            emit(f"Removed {removed} files no longer under {root}")

    indexed = skipped = failed = 0
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
    return {
        "mode": "folder",
        "found": len(files),
        "indexed": indexed,
        "unchanged": skipped,
        "failed": failed,
        "root": str(root),
    }


def _index_one(path: Path, fp: str) -> None:
    extracted = extract(path)
    try:
        ch = content_hash(path)
    except OSError:
        ch = None
    extracted.content_hash = ch
    pieces = []
    card = metadata_card(
        path.name,
        extracted.kind,
        extracted.captured_at,
        extracted.time_source,
        extracted.country,
        extracted.city,
        extracted.language,
        extracted.roles,
        extracted.venues,
        extracted.lat,
        extracted.lon,
        extracted.interpreted,
    )
    pieces.append(card)
    if extracted.interpreted and extracted.interpreted not in extracted.text:
        pieces.append("Interpreted metadata\n" + extracted.interpreted)
    body = normalize(extracted.text)
    content_chunks = chunk_text(body) if body else []
    truncated = False
    if len(content_chunks) > MAX_CHUNKS - 2:
        content_chunks = content_chunks[: MAX_CHUNKS - 2]
        truncated = True
    pieces.extend(content_chunks)
    from local_rag.embedder import embed_texts

    vectors = embed_texts(pieces)
    roles = ",".join(extracted.roles)
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
                "city": extracted.city,
                "roles": roles,
                "vector": np.asarray(vector, dtype=np.float32).tobytes(),
            }
        )
    warnings = list(extracted.warnings)
    if truncated:
        warnings.append(f"kept the first {MAX_CHUNKS} chunks")
    if ch:
        dupes = [p for p in store.files_by_content_hash(ch) if p != str(path)]
        if dupes:
            warnings.append(f"duplicate content also at {Path(dupes[0]).name}")
    preview_source = extracted.interpreted or body or card
    store.replace_file(
        {
            "path": str(path),
            "name": path.name,
            "kind": extracted.kind,
            "fingerprint": fp,
            "captured_at": extracted.captured_at,
            "time_source": extracted.time_source,
            "country": extracted.country,
            "city": extracted.city,
            "language": extracted.language,
            "roles": roles,
            "place_source": extracted.place_source,
            "lat": extracted.lat,
            "lon": extracted.lon,
            "chars": len(body),
            "ocr_used": 1 if extracted.ocr_used else 0,
            "warning": "; ".join(warnings),
            "preview": preview_source[:280],
            "content_hash": ch,
            "enrichment_json": json.dumps(extracted.enrichment or {}, ensure_ascii=False),
            "device": extracted.device,
            "duration": extracted.duration,
        },
        chunk_rows,
    )
