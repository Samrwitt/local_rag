"""Command line: index a folder, ask a question, or start the UI."""

from __future__ import annotations

import argparse
import json

from local_rag.answer import local_preview, retrieve
from local_rag.pipeline import index_folder


def main():
    parser = argparse.ArgumentParser(description="Local folder RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Crawl a folder into the local index")
    index.add_argument("path")
    index.add_argument("--max-files", type=int, default=400)

    ask = sub.add_parser("ask", help="Query the index and print the local preview")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--kind", default=None)
    ask.add_argument("--country", default=None)
    ask.add_argument("--year", default=None)
    ask.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="Start the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    if args.command == "index":
        stats = index_folder(args.path, max_files=args.max_files, on_event=print)
        print(json.dumps(stats))
        return
    if args.command == "ask":
        hits = retrieve(args.question, k=args.k, kind=args.kind, country=args.country, year=args.year)
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
