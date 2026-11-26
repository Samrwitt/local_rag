"""Command line: index, ask (verified), eval, or serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_rag.answer import ask_verified, local_preview
from local_rag.pipeline import index_folder, index_paths
from local_rag.settings import EMBEDDING_CANDIDATES


def main():
    parser = argparse.ArgumentParser(description="Local hybrid retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Index selected files or a folder")
    index.add_argument("path", nargs="?", help="Folder to crawl (scale-up path)")
    index.add_argument("--files", nargs="+", help="Selected file paths (v1)")
    index.add_argument("--max-files", type=int, default=400)
    index.add_argument("--purge-missing", action="store_true")

    ask = sub.add_parser("ask", help="Verified multi-stage retrieval (no LLM)")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--session-id", default=None)
    ask.add_argument("--json", action="store_true")

    ev = sub.add_parser("eval", help="Run retrieval eval against the sample set")
    ev.add_argument("--path", default=None)

    bench = sub.add_parser("benchmark-models", help="Print embedding candidates to benchmark")
    bench.add_argument("--download", action="store_true", help="Attempt to load each candidate")

    serve = sub.add_parser("serve", help="Start the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    if args.command == "index":
        if args.files:
            stats = index_paths(args.files, on_event=print)
        elif args.path:
            stats = index_folder(
                args.path,
                max_files=args.max_files,
                on_event=print,
                purge_missing=args.purge_missing,
            )
        else:
            raise SystemExit("Provide --files or a folder path")
        print(json.dumps(stats))
        return
    if args.command == "ask":
        result = ask_verified(args.question, k=args.k, session_id=args.session_id)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(local_preview(args.question, result=result))
        return
    if args.command == "eval":
        from local_rag.eval_harness import run_eval

        report = run_eval(Path(args.path) if args.path else None)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.command == "benchmark-models":
        print(json.dumps(EMBEDDING_CANDIDATES, indent=2))
        if args.download:
            from transformers import AutoModel, AutoTokenizer

            for key, meta in EMBEDDING_CANDIDATES.items():
                mid = meta["id"]
                try:
                    AutoTokenizer.from_pretrained(mid)
                    AutoModel.from_pretrained(mid)
                    print(f"OK {key}: {mid}")
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {key}: {mid} — {exc}")
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("local_rag.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
