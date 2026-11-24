"""Local retrieval preview, and the OpenAI-compatible request a model would receive."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from local_rag.place import parse_trip_query
from local_rag import store

SYSTEM_PROMPT = (
    "You answer questions about the user's own files — photos, videos, receipts, "
    "hotel info, bookings, and memos. "
    "Use only the numbered sources. Cite the source number like [1]. "
    "If the sources do not contain the answer, say that you cannot tell from these files. "
    "Do not invent countries, dates, or events."
)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        name = Path(hit["path"]).name
        when = hit.get("captured_at") or "time unknown"
        country = hit.get("country") or "country unknown"
        city = hit.get("city") or ""
        place = f"{city}, {country}" if city else country
        roles = hit.get("roles") or ""
        blocks.append(
            f"[{index}] {name} | {hit['kind']} | {when} | {place}"
            + (f" | roles: {roles}" if roles else "")
            + f"\n{hit['text'][:1200]}"
        )
    sources = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nSources:\n{sources}",
        },
    ]


def llm_request_body(question: str, hits: list[dict], model: str) -> dict:
    return {
        "model": model,
        "temperature": 0.2,
        "messages": build_messages(question, hits),
    }


def retrieve(
    question: str,
    k: int = 8,
    kind: str | None = None,
    country: str | None = None,
    year: str | None = None,
    city: str | None = None,
) -> list[dict]:
    from local_rag.embedder import embed_query

    hints = parse_trip_query(question)
    vector = embed_query(question)
    return store.search(
        question,
        vector,
        k=k,
        kind=kind or None,
        country=country or None,
        year=year or None,
        city=city or None,
        trip_hints=hints,
    )


def local_preview(question: str, hits: list[dict]) -> str:
    if not hits:
        return "No matches."
    lines = []
    for index, hit in enumerate(hits, start=1):
        name = Path(hit["path"]).name
        when = hit.get("captured_at") or "—"
        country = hit.get("country") or "—"
        city = hit.get("city")
        place = f"{city}, {country}" if city else country
        roles = hit.get("roles")
        extra = f" · {roles}" if roles else ""
        lines.append(f"{index}. {name} · {hit['kind']} · {when} · {place}{extra} · {hit['score']:.2f}")
        lines.append(hit["text"][:900].strip())
        lines.append("")
    return "\n".join(lines).strip()


def call_llm(base_url: str, model: str, api_key: str, body: dict, timeout: int = 120) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach the LLM at {url}: {exc.reason}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:400]}") from exc
