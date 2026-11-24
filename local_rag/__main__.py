"""Command line: index selected files or a folder, ask, or serve the UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_rag.answer import local_preview, retrieve
from local_rag.pipeline import index_folder, index_paths


def main():
    parser = argparse.ArgumentParser(description="Local folder RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Index selected files or a folder")
    index.add_argument("path", nargs="?", help="Folder to crawl (scale-up path)")
    index.add_argument("--files", nargs="+", help="Selected file paths (v1)")
    index.add_argument("--max-files", type=int, default=400)
    index.add_argument("--purge-missing", action="store_true")

    ask = sub.add_parser("ask", help="Query the index and print the local preview")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=8)
    ask.add_argument("--kind", default=None)
    ask.add_argument("--country", default=None)
    ask.add_argument("--year", default=None)
    ask.add_argument("--city", default=None)
    ask.add_argument("--json", action="store_true")

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
        hits = retrieve(
            args.question,
            k=args.k,
            kind=args.kind,
            country=args.country,
            year=args.year,
            city=args.city,
        )
        if args.json:
            print(json.dumps({"answer": local_preview(args.question, hits), "sources": hits}, indent=2))
        else:
            print(local_preview(args.question, hits))
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("local_rag.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
