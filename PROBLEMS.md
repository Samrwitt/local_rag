# Product rules (locked)

## 1. Access

Do not worry about passwords or permission prompts. Access is treated as **all allowed** for the user who runs the app.

## 2. Indexing path

**Now:** index selected files (and optional folder batches).  
**Later:** scale to auto-crawl and index all possible files within authorization.  
Folder crawl already exists as the scale-up path; selected files stay the default.

## 3. OCR is interpreted, not stored raw

OCR may return cafe names, street lines, and other text. That text is passed through an **interpret** step (`local_rag/place.py`) into:

- language cues  
- country / city  
- document roles (receipt, hotel, booking, memo, …)

Each attribute keeps method, confidence, evidence, and observed vs inferred flags in `enrichment_json`.

## 4. Photos and videos for trip queries

**Videos:** do not open or watch frames / speech. Only container **metadata** (headers/tags) plus the file name are decoded into time/place signals.

**Photos:** OCR + EXIF/GPS when present, then the same place interpretation.

**Goal query:** *“my trip info to Italy in 2019”* should surface related **photo, video, receipt, hotel info, booking info, memo**, etc. — diversified across files (one strong chunk per file first), with soft boosts for matching country / year / city / trip roles.

---

# Practical obstacles (observed)

| Problem | Root cause | Selected fix | Remaining |
|---|---|---|---|
| Italy 2019 query has no true positives | Corpus has almost no Italy-tagged personal trip media | File-eval marks it as expected abstain; do not invent success | Need real trip files or a declared fixture set |
| “hotel”/“receipt” false positives | Role regex matches medical guides / proposals | Travel co-evidence + citation filter in `place.py` | Still leaks on some PDFs; tighten further |
| City lists are finite | Hand-maintained `CITY_COUNTRY` | Extend list; GPS country via polygons | Unknown towns need geocoder |
| Photos without OCR/GPS hard to find by scene | Text-only MiniLM embeddings | Metadata cards + OCR when present | CLIP / visual embeddings not shipped |
| Underscores in names | `Italy_2019_rome.mp4` | Normalize `_`/`-` to spaces in place matching | OK |
| Small generative models invent dates | LLM optional last step | Verified / local passages first; LLM fail-open | Keep LLM off by default |
| Duplicate near-copies | Same content different paths | Content sample hash + warning | Full-file hash optional for tiny files |
| HEIC | Pillow lacks HEIF by default | pillow-heif or heif-convert/magick fallback | Depends on OS packages |
