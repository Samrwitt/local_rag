"""Answer API: multi-stage verified retrieval first; generative LLM optional/last."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

from local_rag.retrieval import retrieve_verified

SYSTEM_PROMPT = (
    "You answer ONLY from the numbered sources about the user's own files. "
    "Cite sources like [1]. "
    "If the sources are missing, weak, or off-topic, reply exactly: "
    "I cannot tell from the indexed files. "
    "Do not use outside knowledge. Do not invent people, places, dates, amounts, or events."
)


def retrieve(
    question: str,
    k: int = 5,
    kind: str | None = None,
    country: str | None = None,
    year: str | None = None,
    city: str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Back-compat wrapper — returns source list from the verified pipeline."""
    _ = (country, year, city)
    result = retrieve_verified(question, session_id=session_id, k=k, kind=kind)
    return result.get("sources") or []


def ask_verified(
    question: str,
    k: int = 5,
    session_id: str | None = None,
    kind: str | None = None,
) -> dict:
    return retrieve_verified(question, session_id=session_id, k=k, kind=kind)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    if not hits:
        sources = "(no confident sources retrieved)"
    else:
        blocks = []
        for index, hit in enumerate(hits, start=1):
            name = hit.get("name") or hit.get("entry_id") or hit.get("id") or f"source-{index}"
            if hit.get("path"):
                name = Path(hit["path"]).name
            when = hit.get("captured_at") or "—"
            place = hit.get("country") or hit.get("category") or "—"
            blocks.append(
                f"[{index}] {name} | {hit.get('source_type') or hit.get('kind') or '—'} | {when} | {place}\n"
                + (hit.get("text") or hit.get("question") or hit.get("answer_en") or "")[:1200]
            )
        sources = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"},
    ]


def llm_request_body(question: str, hits: list[dict], model: str) -> dict:
    return {"model": model, "temperature": 0, "messages": build_messages(question, hits)}


def local_preview(question: str, hits: list[dict] | None = None, result: dict | None = None) -> str:
    result = result or ask_verified(question)
    hits = hits if hits is not None else (result.get("sources") or [])
    lines = [
        f"Answer ({'abstained' if result.get('abstained') else 'verified'} · "
        f"conf {float(result.get('confidence') or 0):.2f} · lang {result.get('language')} · "
        f"intent {result.get('intent') or '—'}):",
        result.get("answer") or "",
        "",
        f"Sources ({len(hits)}):",
    ]
    for index, hit in enumerate(hits, start=1):
        label = hit.get("entry_id") or hit.get("name") or (
            Path(hit["path"]).name if hit.get("path") else hit.get("id")
        )
        kind = hit.get("source_type") or hit.get("kind") or "—"
        conf = hit.get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        lines.append(f"{index}. {label} · {kind} · conf {conf_s}")
        snippet = hit.get("question") or hit.get("text") or ""
        lines.append(snippet[:400].strip())
        lines.append("")
    stages = result.get("stages") or {}
    if stages.get("validation"):
        lines.append(f"Validation: {stages['validation']}")
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
