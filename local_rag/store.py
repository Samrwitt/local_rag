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


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "from", "with",
    "about", "what", "when", "where", "who", "how", "did", "do", "does", "is",
    "are", "was", "were", "my", "me", "i", "you", "your", "our", "this", "that",
    "these", "those", "into", "over", "under", "than", "then", "also", "any",
    "have", "has", "had", "write", "wrote", "written", "tell", "show", "find",
    "please", "can", "could", "would", "should",
}

PLACEISH = {
    "addis", "ababa", "ethiopia", "italy", "italia", "france", "germany", "spain",
    "kenya", "rome", "paris", "london", "berlin", "hotel", "booking", "receipt",
    "trip", "travel", "vacation", "holiday", "visit", "year", "country", "city",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))


def _query_terms(query: str) -> set[str]:
    return {term for term in _tokens(query) if term not in STOPWORDS}


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
    query_terms = _query_terms(query)
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
        distinctive = {term for term in query_terms if term not in PLACEISH}
        content_hits = query_terms & text_terms
        distinct_in_text = distinctive & text_terms
        name_hits = (len(query_terms & path_terms) / len(query_terms)) if query_terms else 0.0
        # Filename matches on place words alone are weak (every Addis tour image).
        if distinctive:
            name_hits = len(distinctive & path_terms) / len(distinctive)
        rare_bonus = 0.0
        for term in distinct_in_text:
            if len(term) >= 4:
                rare_bonus += 0.10
        rare_bonus = min(rare_bonus, 0.30)
        score = 0.60 * cosine + 0.20 * overlap + 0.10 * name_hits + rare_bonus
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
        if len(hits) >= max(k * 3, 12):
            break
    if len(hits) < max(k * 3, 12):
        hits.extend(leftovers[: max(k * 3, 12) - len(hits)])
    # Prefer files tagged with the asked country; if none, only keep strong
    # non-place evidence (so "Italy hotel" with no Italy files returns nothing).
    if effective_country:
        country_l = str(effective_country).lower()
        matched = [
            hit
            for hit in hits
            if (hit.get("country") or "").lower() == country_l
            or country_l in Path(hit["path"]).name.lower()
            or country_l in (hit.get("text") or "").lower()
        ]
        if matched:
            hits = matched
        else:
            distinctive = {term for term in query_terms if term not in PLACEISH}
            strong = []
            for hit in hits:
                terms = _tokens(hit.get("text") or "") | _tokens(Path(hit["path"]).name)
                if distinctive and (distinctive & terms) and hit["score"] >= 0.55:
                    strong.append(hit)
            hits = strong
    return _keep_confident(hits, k=k, query_terms=query_terms)


def _keep_confident(hits: list[dict], k: int, query_terms: set[str] | None = None) -> list[dict]:
    """Drop near-misses so weak files never reach the answer step."""
    if not hits:
        return []
    top = hits[0]
    # Absolute floor: below this, treat as "no good match".
    if top["score"] < 0.48 or top["cosine"] < 0.32:
        return []
    distinctive = {term for term in (query_terms or set()) if term not in PLACEISH}
    top_text = _tokens(top.get("text") or "")
    top_has_distinct = bool(distinctive & top_text) if distinctive else False
    kept = []
    for hit in hits:
        if hit["score"] < max(0.45, top["score"] * 0.82):
            continue
        if hit["cosine"] < 0.25 and hit["overlap"] < 0.4:
            continue
        # If the best hit matched a distinctive word, later hits must too.
        if top_has_distinct and distinctive:
            hit_terms = _tokens(hit.get("text") or "") | _tokens(Path(hit["path"]).name)
            if not (distinctive & hit_terms):
                continue
        kept.append(hit)
        if len(kept) >= k:
            break
    return kept
