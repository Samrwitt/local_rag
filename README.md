# Local files — trip-aware personal search

Index files on this computer, interpret place/time, then search.  
Access is **all allowed** for the running user.

## Product rules

1. **Access** — no password/permission gate; treat the host user’s files as allowed.  
2. **Indexing** — start with **selected files**; later **auto-crawl** everything within authorization.  
3. **OCR** — not stored raw. Cafe/street text is **interpreted** into language, country, city, and roles.  
4. **Photos & videos** — photos use OCR/EXIF; videos use **metadata only** (no frame/speech). A query like *“my trip info to Italy in 2019”* should return related photo, video, receipt, hotel, booking, memo, etc.

## Run

```bash
cd /home/lenovo/local_rag
python3 -m local_rag serve
```

Open http://127.0.0.1:8765

```bash
python3 -m local_rag index --files /path/to/a.pdf /path/to/b.jpg
python3 -m local_rag ask "my trip info to Italy in 2019"
```

## How indexing works

| Kind | What we take | What we store |
|---|---|---|
| Text / PDF / DOCX | Text layer | Chunks + interpreted place/time when present |
| Photo | OCR + EXIF/GPS | Interpreted country/city/language/roles + searchable card |
| Video | Container metadata + filename only | Same trip fields — **no** frame reading |

Place interpretation: `local_rag/place.py`  
Video extract: metadata-only path in `local_rag/extract.py`  
Selected vs folder crawl: `local_rag/pipeline.py`

## Retrieval (no generative LLM required)

```
question
  → preprocess
  → dense + BM25
  → RRF hybrid
  → optional rerank
  → confidence / abstain
  → verified answer (curated KB and/or indexed files)
```

Trip search keeps **one strong chunk per file** first so Italy 2019 can surface receipt + hotel + video + memo together.

## Formats

Text, PDF, DOCX, PPTX, XLSX, common images (incl. HEIC when tools available), common videos (metadata only).

Indexing uses content hashing for duplicates, skips unchanged files, and continues when one file fails.

## Optional extras

- Curated Am/En QA KB: `data/kb/curated_qa.json`  
- KB eval: `python3 -m local_rag eval`  
- **Real-file search eval:** `python3 -m local_rag file-eval`  
- Cross-file links: `python3 -m local_rag relations`  
- Embedding candidates / reranker: see `local_rag/settings.py`  
- Audit notes: `AUDIT.md` / `PROBLEMS.md`  

