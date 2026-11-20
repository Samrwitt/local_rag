"""SQLite index of chunks and their vectors."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np

from local_rag.settings import MODEL_ID

INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "index.sqlite"


def connect() -> sqlite3.Connection:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            name TEXT,
            kind TEXT,
            fingerprint TEXT,
            captured_at TEXT,
            time_source TEXT,
            country TEXT,
            lat REAL,
            lon REAL,
            chars INTEGER,
            ocr_used INTEGER,
            warning TEXT,
            preview TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            path TEXT,
            ordinal INTEGER,
            text TEXT,
            kind TEXT,
            captured_at TEXT,
            country TEXT,
            vector BLOB
        );
        """
    )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('model', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (MODEL_ID,),
    )
    conn.commit()
    return conn


def indexed_fingerprint(path: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT fingerprint FROM files WHERE path = ?", (path,)).fetchone()
    return row["fingerprint"] if row else None


def replace_file(record: dict, chunks: list[dict]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE path = ?", (record["path"],))
        conn.execute(
            """
            INSERT INTO files(path, name, kind, fingerprint, captured_at, time_source, country, lat, lon, chars, ocr_used, warning, preview)
            VALUES (:path, :name, :kind, :fingerprint, :captured_at, :time_source, :country, :lat, :lon, :chars, :ocr_used, :warning, :preview)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                kind=excluded.kind,
                fingerprint=excluded.fingerprint,
                captured_at=excluded.captured_at,
                time_source=excluded.time_source,
                country=excluded.country,
                lat=excluded.lat,
                lon=excluded.lon,
                chars=excluded.chars,
                ocr_used=excluded.ocr_used,
                warning=excluded.warning,
                preview=excluded.preview
            """,
            record,
        )
        conn.executemany(
            """
            INSERT INTO chunks(path, ordinal, text, kind, captured_at, country, vector)
            VALUES (:path, :ordinal, :text, :kind, :captured_at, :country, :vector)
            """,
            chunks,
        )


def delete_missing(keep_paths: set[str]) -> int:
    with connect() as conn:
        rows = conn.execute("SELECT path FROM files").fetchall()
        stale = [row["path"] for row in rows if row["path"] not in keep_paths]
        for path in stale:
            conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            conn.execute("DELETE FROM files WHERE path = ?", (path,))
    return len(stale)


def list_files() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT path, name, kind, captured_at, time_source, country, lat, lon, chars, ocr_used, warning, preview FROM files ORDER BY name"
        ).fetchall()
    return [dict(row) for row in rows]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))


def search(
    query: str,
    query_vec: np.ndarray,
    k: int = 5,
    kind: str | None = None,
    country: str | None = None,
    year: str | None = None,
) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, path, ordinal, text, kind, captured_at, country, vector FROM chunks"
        ).fetchall()
    if not rows:
        return []
    query_terms = _tokens(query)
    scored = []
    qvec = query_vec.astype(np.float32)
    for row in rows:
        if kind and row["kind"] != kind:
            continue
        if country and (row["country"] or "").lower() != country.lower():
            continue
        if year and not (row["captured_at"] or "").startswith(str(year)):
            continue
        blob = row["vector"]
        if not blob:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        cosine = float(vector @ qvec)
        terms = _tokens(row["text"])
        overlap = (len(query_terms & terms) / len(query_terms)) if query_terms else 0.0
        # Cosine finds paraphrases. Term overlap keeps rare names like "eder" or "whisper".
        score = 0.75 * cosine + 0.25 * overlap
        scored.append((score, cosine, overlap, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits = []
    for score, cosine, overlap, row in scored[:k]:
        hits.append(
            {
                "path": row["path"],
                "ordinal": row["ordinal"],
                "text": row["text"],
                "kind": row["kind"],
                "captured_at": row["captured_at"],
                "country": row["country"],
                "score": score,
                "cosine": cosine,
                "overlap": overlap,
            }
        )
    return hits
