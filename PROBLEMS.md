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

It is **not** “as-is” OCR sitting unused in the database. The searchable card carries the interpreted place and roles.

## 4. Photos and videos for trip queries

**Videos:** do not open or watch frames / speech. Only container **metadata** (headers/tags) plus the file name are decoded into time/place signals.

**Photos:** OCR + EXIF/GPS when present, then the same place interpretation.

**Goal query:** *“my trip info to Italy in 2019”* should surface related **photo, video, receipt, hotel info, booking info, memo**, etc. — diversified across files (one strong chunk per file first), with soft boosts for matching country / year / city / trip roles.

---

# Practical obstacles

- City lists are finite; unknown towns need list growth or a geocoder later.  
- Underscores in names (`Italy_2019_rome.mp4`) are normalized to spaces for matching.  
- Weak place words (conference cities, bare “note”) must not invent fake trips.  
- Small generative models can invent dates; verified / local passages come first.  
