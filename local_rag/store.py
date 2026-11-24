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
            city TEXT,
            language TEXT,
            roles TEXT,
            place_source TEXT,
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
            city TEXT,
            roles TEXT,
            vector BLOB
        );
        """
    )
    _ensure_columns(
        conn,
        "files",
        {
            "city": "TEXT",
            "language": "TEXT",
            "roles": "TEXT",
            "place_source": "TEXT",
        },
    )
    _ensure_columns(conn, "chunks", {"city": "TEXT", "roles": "TEXT"})
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('model', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (MODEL_ID,),
    )
    conn.commit()
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, typ in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")


def indexed_fingerprint(path: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT fingerprint FROM files WHERE path = ?", (path,)).fetchone()
    return row["fingerprint"] if row else None


def replace_file(record: dict, chunks: list[dict]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE path = ?", (record["path"],))
        conn.execute(
            """
            INSERT INTO files(path, name, kind, fingerprint, captured_at, time_source, country, city, language, roles, place_source, lat, lon, chars, ocr_used, warning, preview)
            VALUES (:path, :name, :kind, :fingerprint, :captured_at, :time_source, :country, :city, :language, :roles, :place_source, :lat, :lon, :chars, :ocr_used, :warning, :preview)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                kind=excluded.kind,
                fingerprint=excluded.fingerprint,
                captured_at=excluded.captured_at,
                time_source=excluded.time_source,
                country=excluded.country,
                city=excluded.city,
                language=excluded.language,
                roles=excluded.roles,
                place_source=excluded.place_source,
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
            INSERT INTO chunks(path, ordinal, text, kind, captured_at, country, city, roles, vector)
            VALUES (:path, :ordinal, :text, :kind, :captured_at, :country, :city, :roles, :vector)
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
            """
            SELECT path, name, kind, captured_at, time_source, country, city, language, roles,
                   place_source, lat, lon, chars, ocr_used, warning, preview
            FROM files ORDER BY name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))


def search(
    query: str,
    query_vec: np.ndarray,
    k: int = 8,
    kind: str | None = None,
    country: str | None = None,
    year: str | None = None,
    city: str | None = None,
    trip_hints: dict | None = None,
) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, path, ordinal, text, kind, captured_at, country, city, roles, vector FROM chunks"
        ).fetchall()
    if not rows:
        return []
    hints = trip_hints or {}
    effective_country = country or hints.get("country")
    effective_year = year or hints.get("year")
    effective_city = city or hints.get("city")
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
        path_name = Path(row["path"]).name.lower()
        path_terms = _tokens(path_name)
        text_terms = _tokens(row["text"])
        terms = text_terms | path_terms
        overlap = (len(query_terms & terms) / len(query_terms)) if query_terms else 0.0
        # Rare content words (eder, displacement) matter more than place names in the filename.
        content_hits = query_terms & text_terms
        name_hits = (len(query_terms & path_terms) / len(query_terms)) if query_terms else 0.0
        rare_bonus = 0.0
        for term in content_hits:
            if len(term) >= 4 and term not in {"addis", "ababa", "ethiopia", "italy", "france", "what", "write", "about", "with", "from", "this", "that", "have", "trip", "travel"}:
                rare_bonus += 0.08
        rare_bonus = min(rare_bonus, 0.24)
        score = 0.62 * cosine + 0.22 * overlap + 0.12 * name_hits + rare_bonus
        # Soft trip boosts so "Italy 2019" lifts photos, videos, receipts together.
        # Only boost when the file actually carries that metadata — not every
        # academic paper that once mentioned a conference city.
        if effective_country and (row["country"] or "").lower() == str(effective_country).lower():
            score += 0.10
        if effective_year:
            if (row["captured_at"] or "").startswith(str(effective_year)):
                score += 0.10
            elif str(effective_year) in path_name:
                score += 0.08
        if effective_city and (row["city"] or "").lower() == str(effective_city).lower():
            score += 0.06
        if hints.get("trip") and row["roles"]:
            # Only boost real trip roles when the place also matches the query.
            roles = {part.strip() for part in (row["roles"] or "").split(",") if part.strip()}
            if roles & {"receipt", "hotel", "booking", "memo"}:
                if effective_country and (row["country"] or "").lower() == str(effective_country).lower():
                    score += 0.05
                elif not effective_country and not effective_city:
                    score += 0.03
        scored.append((score, cosine, overlap, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    # One strong chunk per file first, so a trip query returns receipt + hotel + video + memo.
    hits = []
    seen_paths = set()
    leftovers = []
    for score, cosine, overlap, row in scored:
        item = {
            "path": row["path"],
            "ordinal": row["ordinal"],
            "text": row["text"],
            "kind": row["kind"],
            "captured_at": row["captured_at"],
            "country": row["country"],
            "city": row["city"],
            "roles": row["roles"],
            "score": score,
            "cosine": cosine,
            "overlap": overlap,
        }
        if row["path"] not in seen_paths:
            hits.append(item)
            seen_paths.add(row["path"])
        else:
            leftovers.append(item)
        if len(hits) >= k:
            break
    if len(hits) < k:
        hits.extend(leftovers[: k - len(hits)])
    return hits
