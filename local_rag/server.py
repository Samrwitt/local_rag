"""Local web UI and JSON API. Access is treated as allowed for this host user."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from local_rag.answer import ask_verified, call_llm, llm_request_body, local_preview
from local_rag.browse import list_dir, roots
from local_rag.pipeline import index_folder, index_paths
from local_rag.settings import EMBEDDING_CANDIDATES, MODEL_ID, RERANKER_CANDIDATES
from local_rag import kb, store

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static" / "index.html"

app = FastAPI(title="Local files", version="0.2.0")

_job_lock = threading.Lock()
_job = {
    "state": "idle",
    "log": [],
    "stats": None,
    "error": None,
}


class IndexRequest(BaseModel):
    # Selected files for v1. Folder crawl is the scale-up path.
    paths: list[str] = Field(default_factory=list)
    path: str | None = None
    max_files: int = Field(default=400, ge=1, le=5000)
    purge_missing: bool = False


class QueryRequest(BaseModel):
    question: str
    mode: str = "local"  # local = verified retrieval; llm = optional last step
    k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    kind: str | None = None
    country: str | None = None
    year: str | None = None
    city: str | None = None
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:1.5b"
    llm_api_key: str = ""


def _log(message: str) -> None:
    with _job_lock:
        _job["log"].append(message)
        _job["log"] = _job["log"][-200:]


@app.get("/")
def home():
    return FileResponse(STATIC)


@app.get("/api/info")
def info():
    files = store.list_files()
    kb.load_entries()
    return {
        "embedding_model": MODEL_ID,
        "embedding_candidates": EMBEDDING_CANDIDATES,
        "reranker_candidates": RERANKER_CANDIDATES,
        "retrieval": "preprocess → dense + BM25 → RRF → optional rerank → confidence",
        "kb_entries": len(kb.load_entries()),
        "ocr_language": "eng+am possible later",
        "indexed_files": len(files),
        "access": "allowed for the running user; selected files now, auto-crawl later",
        "llm_api": "optional final step only; default path is verified retrieval",
        "default_llm_base_url": "http://127.0.0.1:11434/v1",
    }


@app.get("/api/documents")
def documents():
    return {"files": store.list_files()}


@app.get("/api/browse/roots")
def browse_roots():
    return {"roots": roots()}


@app.get("/api/browse")
def browse(path: str | None = None):
    try:
        return list_dir(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/job")
def job():
    with _job_lock:
        return dict(_job)


@app.post("/api/index")
def start_index(body: IndexRequest):
    paths = [p.strip() for p in body.paths if p and p.strip()]
    folder = body.path.strip() if body.path else None
    if not paths and not folder:
        raise HTTPException(status_code=400, detail="Provide selected file paths, or a folder path")

    with _job_lock:
        if _job["state"] == "running":
            raise HTTPException(status_code=409, detail="An index is already running")
        _job["state"] = "running"
        _job["log"] = ["Starting index"]
        _job["stats"] = None
        _job["error"] = None

    def run():
        try:
            if paths:
                stats = index_paths(paths, on_event=_log)
            else:
                stats = index_folder(
                    folder,
                    max_files=body.max_files,
                    on_event=_log,
                    purge_missing=body.purge_missing,
                )
            with _job_lock:
                _job["state"] = "done"
                _job["stats"] = stats
        except Exception as exc:
            _log(str(exc))
            with _job_lock:
                _job["state"] = "error"
                _job["error"] = str(exc)

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


@app.post("/api/query")
def query(body: QueryRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    kind = None if not body.kind or body.kind == "any" else body.kind
    result = ask_verified(question, k=body.k, session_id=body.session_id, kind=kind)
    hits = result.get("sources") or []
    request_body = llm_request_body(question, hits, body.llm_model)
    payload = {
        "mode": body.mode,
        "question": question,
        "resolved_question": result.get("resolved_question"),
        "session_id": result.get("session_id"),
        "language": result.get("language"),
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "abstained": result.get("abstained"),
        "stages": result.get("stages"),
        "sources": hits,
        "llm_used": False,
        "llm_request": {
            "method": "POST",
            "url": body.llm_base_url.rstrip("/") + "/chat/completions",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer <key, omitted if empty>"},
            "body": request_body,
        },
        "answer": None,
        "llm_error": None,
    }
    if body.mode != "llm":
        payload["answer"] = result.get("answer")
        payload["preview"] = local_preview(question, hits, result=result)
        return payload
    if result.get("abstained") or not hits:
        payload["answer"] = result.get("answer")
        return payload
    try:
        payload["answer"] = call_llm(body.llm_base_url, body.llm_model, body.llm_api_key, request_body)
        payload["llm_used"] = True
    except Exception as exc:
        payload["llm_error"] = str(exc)
        payload["answer"] = result.get("answer")
    return payload
