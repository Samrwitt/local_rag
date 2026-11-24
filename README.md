# Local files

Index files on this computer, then search them.

Selected files first. Folder crawl later. Videos use metadata only. OCR/text is interpreted into place and document roles when possible.

```bash
cd /home/lenovo/local_rag
python3 -m local_rag serve
```

Open http://127.0.0.1:8765

```bash
python3 -m local_rag index --files /path/to/a.pdf /path/to/b.md
python3 -m local_rag ask "your question"
```

The index is `data/index.sqlite` (gitignored).
