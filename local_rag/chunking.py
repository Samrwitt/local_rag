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
                chunks.append(para[start : start + size].strip())
                if start + size >= len(para):
                    break
                start += size - overlap
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > size:
            chunks.append(current)
            tail = current[-overlap:].strip()
            current = f"{tail}\n\n{para}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def metadata_card(path_name: str, kind: str, captured_at: str | None, time_source: str | None, country: str | None, lat: float | None, lon: float | None) -> str:
    lines = [f"File: {path_name}", f"Type: {kind}"]
    if captured_at:
        lines.append(f"Time: {captured_at} (from {time_source})")
    else:
        lines.append("Time: unknown")
    if country:
        lines.append(f"Country: {country}")
    else:
        lines.append("Country: unknown")
    if lat is not None and lon is not None:
        lines.append(f"Coordinates: {lat:.5f}, {lon:.5f}")
    return "\n".join(lines)
