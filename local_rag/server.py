"""Small local web UI and JSON API."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from local_rag.answer import call_llm, llm_request_body, local_preview, retrieve
from local_rag.settings import MODEL_ID
from local_rag.pipeline import index_folder
from local_rag import store

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static" / "index.html"

app = FastAPI(title="Local RAG", version="0.1.0")

_job_lock = threading.Lock()
_job = {
    "state": "idle",
    "log": [],
    "stats": None,
    "error": None,
}


class IndexRequest(BaseModel):
    path: str
    max_files: int = Field(default=400, ge=1, le=5000)


class QueryRequest(BaseModel):
    question: str
    mode: str = "local"
    k: int = Field(default=5, ge=1, le=12)
    kind: str | None = None
    country: str | None = None
    year: str | None = None
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
    return {
        "embedding_model": MODEL_ID,
        "ocr_language": "eng",
        "indexed_files": len(files),
        "llm_api": "POST {base}/chat/completions  (OpenAI-compatible)",
        "default_llm_base_url": "http://127.0.0.1:11434/v1",
    }


@app.get("/api/documents")
def documents():
    return {"files": store.list_files()}


@app.get("/api/job")
def job():
    with _job_lock:
        return dict(_job)


@app.post("/api/index")
def start_index(body: IndexRequest):
    folder = Path(body.path).expanduser()
    with _job_lock:
        if _job["state"] == "running":
            raise HTTPException(status_code=409, detail="An index is already running")
        _job["state"] = "running"
        _job["log"] = [f"Starting index of {folder}"]
        _job["stats"] = None
        _job["error"] = None

    def run():
        try:
            stats = index_folder(folder, max_files=body.max_files, on_event=_log)
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
    hits = retrieve(
        question,
        k=body.k,
        kind=None if not body.kind or body.kind == "any" else body.kind,
        country=body.country.strip() if body.country else None,
        year=body.year.strip() if body.year else None,
    )
    request_body = llm_request_body(question, hits, body.llm_model)
    payload = {
        "mode": body.mode,
        "question": question,
        "sources": hits,
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
        payload["answer"] = local_preview(question, hits)
        return payload
    try:
        payload["answer"] = call_llm(body.llm_base_url, body.llm_model, body.llm_api_key, request_body)
    except Exception as exc:
        payload["llm_error"] = str(exc)
        payload["answer"] = local_preview(question, hits)
    return payload
