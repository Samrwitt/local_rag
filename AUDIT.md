# Audit & improvement log — multimodal personal file search

Date: 2026-11-26  
Corpus under test: **427 indexed files / 5043 chunks** on this machine (`data/index.sqlite`).

## Baseline (before this pass)

| Area | Status | Notes |
|---|---|---|
| Selected-file + folder crawl indexing | Working | Per-file failures isolated; mtime/size fingerprint |
| PDF / DOCX / TXT / images / video metadata | Working | Video: ffprobe only (no frames) |
| PPTX / XLSX / HEIC | Missing | Added this pass |
| Content hashing / duplicates | Partial | Size+mtime only → now content sample hash |
| Place interpretation | Working | City/country/roles; false positives on “hotel” in guides |
| Evidence + confidence on attributes | Missing | Added `enrichment_json` + `evidence.py` |
| Cross-file relations | Missing | Added `relations.py` + SQLite table |
| Hybrid retrieval (dense+BM25+RRF) | Working | KB-oriented eval was strong; file search weaker |
| Match explanations | Missing | Added `explain.py` + UI chips |
| Image embeddings (CLIP) | Missing | Still text embeddings of OCR/cards only |
| Reverse geocoding beyond country polygons | Partial | `geo.country_at` only |
| Local mode without LLM | Working | Default path |
| Optional LLM | Working | Fail-open to verified answer |
| UI | Shelf (baby green) | Kept (prior Drive redesign rejected); search-first + filters |
| Italy 2019 trip in corpus | **Not present** | Cannot claim trip-retrieval success for that query |

### KB eval baseline (curated QA, not personal files)

Ran `python3 -m local_rag eval`: Am/En menstrual KB items retrieved with high confidence; OOD weather/crypto abstained; emergency escalated. This measures the curated KB path, **not** personal-file trip search.

## Changes in this pass

1. **Ingestion** — PPTX / XLSX extractors (OOXML, no new deps); HEIC via pillow-heif / heif-convert; content hash; enrichment + device/duration columns; extractor rev 10.
2. **Enrichment** — every place/role attribute can carry method, confidence, observed/inferred, evidence (`evidence.py` + `place._enrichment_bundle`).
3. **Relations** — confidence-based pairs (GPS, city, folder, roles, time); year-alone links rejected; CLI `python3 -m local_rag relations`.
4. **Retrieval** — trip hints, file diversification, template answers, match reasons, related-file expansion; filters wired through API.
5. **Eval** — `data/eval/file_search_questions.json` + `python3 -m local_rag file-eval` against **real** indexed files.
6. **UI** — search first; type/year filters; why-matched chips; open file + image preview; dropdown-only add flow preserved.

## Honest testing limitations

- **No Italy 2019 trip set** in the indexed personal files. The file-eval item `italy_2019_missing` expects abstain / no fabricated Italy trip — not Precision@5 success.
- Role patterns still over-tag some academic PDFs as `receipt`/`hotel`; needs stricter travel co-evidence (tracked below).
- No CLIP / visual embedding benchmark yet — photos without OCR remain weak for “pictures of mountains”.
- Full reindex of 427 files after extractor rev was **not** forced (incremental fingerprints change on next touch); new fields populate on re-index.

## Remaining priorities

1. Stricter role tagging (require travel/path co-evidence).
2. Optional CLIP text-to-image path + model benchmark table.
3. Multilingual OCR (`eng+am` / `eng+ita`) when tessdata present.
4. Auto-crawl scheduler (architecture already supports `index_folder`).
5. Relationship rebuild after each index batch in the server job.

## File-search eval (real index, post-fix)

Command: `python3 -m local_rag file-eval` on 427 indexed local files.

| Metric | Value | Target | Pass |
|---|---|---|---|
| Precision@5 | **0.74** | 0.50 | yes |
| MRR | **1.00** | 0.50 | yes |
| Italy-2019 abstain | **1.00** | 0.80 | yes |
| Avg latency | ~7s | — | — |

Italy 2019 trip media was **not** in the corpus; that item measures abstain behavior, not trip recall.

