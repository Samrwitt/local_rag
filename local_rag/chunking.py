"""Whitespace cleanup and paragraph-aware chunks."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_start(text: str, index: int) -> int:
    if index <= 0 or index >= len(text):
        return index
    if text[index - 1].isspace():
        return index
    space = text.rfind(" ", max(0, index - 80), index)
    if space == -1:
        return index
    return space + 1


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = min(len(para), start + size)
                if end < len(para):
                    end = _word_start(para, end)
                    if end <= start:
                        end = min(len(para), start + size)
                chunks.append(para[start:end].strip())
                if end >= len(para):
                    break
                start = _word_start(para, max(start + 1, end - overlap))
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > size:
            chunks.append(current)
            cut = _word_start(current, max(0, len(current) - overlap))
            tail = current[cut:].strip()
            current = f"{tail}\n\n{para}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def metadata_card(
    path_name: str,
    kind: str,
    captured_at: str | None,
    time_source: str | None,
    country: str | None,
    city: str | None,
    language: str | None,
    roles: list[str] | None,
    venues: list[str] | None,
    lat: float | None,
    lon: float | None,
    interpreted: str = "",
) -> str:
    """Searchable card so trip queries find photos, videos, receipts, bookings."""
    year = (captured_at or "")[:4] if captured_at and len(captured_at) >= 4 else None
    lines = [f"File: {path_name}", f"Type: {kind}"]
    if captured_at:
        lines.append(f"Time: {captured_at} (from {time_source})")
        if year and year.isdigit():
            lines.append(f"Year: {year}")
    else:
        lines.append("Time: unknown")
    if country:
        lines.append(f"Country: {country}")
    else:
        lines.append("Country: unknown")
    if city:
        lines.append(f"City: {city}")
    if language:
        lines.append(f"Language: {language}")
    if venues:
        lines.append("Venues: " + ", ".join(venues))
    if roles:
        lines.append("Roles: " + ", ".join(roles))
    if lat is not None and lon is not None:
        lines.append(f"Coordinates: {lat:.5f}, {lon:.5f}")
    if interpreted:
        lines.append("Interpreted place:")
        lines.append(interpreted)
    # Only hint trips when place came from a real signal and year is known.
    # Do not inject generic "hotel booking memo" words — that pollutes search.
    if country and year and roles:
        lines.append(f"Trip hint: {country} {year}")
    return "\n".join(lines)
