# What got in the way

These are the obstacles that showed up while indexing real files on this machine, and the options for each. The choice in the code is the first one.

The folder used for the trial is `corpus/` (not committed). It points at notes, a DOCX of invoice-parsing code, a German PDF invoice, two camera photos, a 10-second video, and a chart screenshot saved both as PNG and as a textless PDF.

## 1. Crawl

**The last project never opened a folder.** It was a platform: Postgres, Redis, Qdrant, MinIO, Celery, and hosted API keys. A person with notes and photos could not try it.

- Chosen: one process, one SQLite file, a folder path.
- Other options: keep the platform and only add a "local folder" connector (still needs Docker); or upload files one by one in the browser (worse for a whole folder, and photos leave the disk).

**Indexing `/` or the whole home directory** would OCR thousands of files on a machine that is already 94% full.

- Chosen: refuse `/` and the home directory, skip files over 80 MB, stop at 400 files.
- Other options: a dry-run that only lists files; a cost estimate before OCR starts.

**A second click on Index spent ~10 seconds importing torch** even when every file was unchanged.

- Chosen: the embedding library loads only when a file actually needs vectors. After that change, a second index of this corpus finished in 0.6 seconds and reported 9 unchanged.
- Other options: keep a long-running worker so the model stays hot (what the old Celery design was for). Worth it once the corpus is large. Not worth it to answer one question.

**Chunk overlap started mid-word.** A passage was shown as "attered across the outskirts" because the cut landed inside "scattered".

- Chosen: move the cut back to the previous space. `EXTRACTOR_REV` is 3 so the next index rewrites the chunks.
- Other options: sentence tokenization (cleaner, and it splits badly on OCR and on Amharic if a tokenizer is not installed).

## 2. Clean, OCR, time, country

**Telegram left a GPS tag that is only a version byte.** Four photos in `Downloads/Telegram Desktop` looked geotagged. Three of them have `GPSVersionID` and no latitude. Trusting "has a GPS block" would have invented a location.

- Chosen: country is set only when latitude and longitude are both present, then checked against an offline country polygon. The stripped photo is indexed with country blank and a warning.
- Other options: guess the country from the caption or the filename (this would have marked the Addis Ababa *essay* as a photo taken in Ethiopia); call a geocoding API (needs network, and sends coordinates off the machine).

**The photo that does have coordinates is in Ethiopia.** GPS reads 9.00501 N, 38.78258 E. The polygon lookup returns Ethiopia. The same function returns Kenya for Nairobi, France for Paris, and nothing for a point in the ocean.

- Chosen: ray-casting on `world.geo.json` (about 250 KB, already in the repo). No network at index time.
- Other options: bounding boxes (fast, and wrong wherever boxes overlap — the United States box covers Canada); a finer Natural Earth 10m file (better coasts, larger file); a geocoder.

Coastlines and borders in this file are simplified. A photo standing on a border can land in the neighboring country. Addis Ababa is inland, so this trial is not that case.

**The camera clock and the GPS clock disagree by the timezone.** On `ethiopia-photo.jpg` the EXIF time is 14:50:07 and the GPS time is 11:50:02. The offset tag is `+03:00`. On the other photo, Photoshop rewrote `DateTime` to 2 July while `DateTimeOriginal` is still 28 June, with offset `+01:00`.

- Chosen: prefer `DateTimeOriginal`, and append the offset when the camera stored one. The Ethiopia photo is stored as `2026-08-08T14:50:07+03:00`.
- Other options: always store UTC (better for sorting across timezones, worse for "what time of day was this"); always store the file's modified time (wrong after a copy).

Video container time and PDF creation time do not carry a timezone in a reliable way. They are stored as recorded. The chart PDF's creation time is the moment ImageMagick wrote it, not the day of the screenshot. The PNG beside it keeps the screenshot's modified time, 19 May 2026.

**OCR of a chart is half real and half noise.** Tesseract (English only — Amharic is not installed) read the titles "Cooperative (simple_spread)" and "Competitive (simple_tag)" from the screenshot, plus broken legend tokens. The same screenshot wrapped as a PDF had no text layer, so the PDF path rasterized it and OCRed it. A scene photo produced no usable words; that OCR was dropped, and the photo is still findable by its metadata card.

- Chosen: keep OCR only when there are at least five real words. Always store a metadata card so an empty OCR result does not make the file invisible.
- Other options: a word-list filter (drops rare names like "eder" if applied too hard); a vision-language model (heavier, and this machine has no NVIDIA GPU); Amharic traineddata for Tesseract (the right next step if the folder is Amharic, and it is a download).

**Video speech is not transcribed.** `clip.mp4` is 10 seconds, created `2026-11-15T19:14:29`. Three frames were sampled. None had readable text. Whisper is not installed, and the disk has about 32 GB free.

- Chosen: store duration, container time, and frame text if any. The chunk says speech was not transcribed.
- Other options: Whisper tiny (works on CPU, slow, another model download); send the audio to a hosted API (leaves the machine).

**DOCX text is in XML runs, not in the visible paragraph.** A naive tag strip glued words together. The invoice-model DOCX is 5042 characters of real Python after reading `w:t` nodes per paragraph.

## 3. Vectorize

**`sentence-transformers` is not installed. The MiniLM weights are already in the Hugging Face cache, and torch is already installed.** Downloading another embedding stack was the wrong move on a full disk.

- Chosen: mean-pool `sentence-transformers/all-MiniLM-L6-v2` with the installed `transformers`. Vectors are L2-normalized. Search is a dot product, which is cosine.
- Other options: the multilingual MiniLM that is also cached (better if the folder mixes languages, slower on CPU); Ollama `bge-m3` (already downloaded under `/usr/share/ollama`, but the Ollama server was not running); a hosted embedding API (nothing works offline).

**Semantic search ranks a glossary next to the real sentence.** "Which speech recognition models will be compared?" returned the methodology's acronym list first (score 0.44) and the sentence "test whisper v wav2vec2" second (score 0.36). Both are on screen. The acronym list really does say "speech recognition". It does not answer the question.

- Chosen: blend cosine (0.75) with the fraction of query words that appear in the chunk (0.25), and show several passages instead of one.
- Other options: a cross-encoder reranker (more accurate, another model); a stricter lexical requirement (hurts paraphrases).

**MiniLM is English-leaning.** An Amharic note would embed poorly with this model. The multilingual weights are the switch if that becomes the corpus. Tesseract would still need Amharic traineddata before OCR could feed it.

## 4. Ask, locally, before any API exists

**A generated answer that cannot be checked is how the last attempt felt unfinished.** Retrieval itself is the result.

- Chosen: local preview prints the passages, the file, the time, the country, and the score. It never calls a model. The UI also shows the exact JSON that *would* be sent.
- Other options: an extractive sentence highlighter; hiding scores (looks cleaner, hides weak matches). Weak matches under 0.35 are labeled as weak.

Trial questions, local preview, no model:

| Question | Top file | What came back |
| --- | --- | --- |
| What happened to the eder when residents were displaced in Addis Ababa? | `corridor.md` | The eder, the trumpeter, and the line that the eder system cannot be maintained after the displacement |
| Which speech recognition models will be compared? | `methodology.md` | The acronym list, then "whisper v wav2vec2" |
| Who is the customer on the German invoice and what is the VAT number? | `invoice.pdf` | Musterkunde AG, Mr. John Doe, VAT DE199378386, invoice 123100401 |
| Show photos taken in Ethiopia | `ethiopia-photo.jpg` | Country Ethiopia, 9.00501, 38.78258, 8 Aug 2026 14:50 +03:00. The photo with stripped GPS scored 0.29 and is labeled country unknown |
| What video files are in the folder and when were they created? | `clip.mp4` | 15 Nov 2026 19:14, 10 seconds, no on-screen text |

## 5. The language model call

`ollama list` from a sandboxed shell failed with "could not connect". The server was actually running. The model files live under `/usr/share/ollama`, and `GET http://127.0.0.1:11434/api/tags` succeeds from this machine. Models present include `qwen2:0.5b`, `qwen2.5:1.5b`, `llama3.2:3b`, `qwen3:4b-instruct`, and `bge-m3`.

The same eder question was sent through the UI.

- `qwen2:0.5b` answered in a few seconds and invented a date: it said the eder was replaced by corridor development "in the 1980s". That year is not in the note.
- `qwen2.5:1.5b` took about 80 seconds and stayed on the sources: the eder could not be maintained once members were scattered, and new neighbors were less dedicated. It did not add a year.

- Chosen: speak the OpenAI chat-completions shape (`POST {base}/chat/completions`). Default base `http://127.0.0.1:11434/v1`, default model `qwen2.5:1.5b`, key optional. If the call fails, the same local passages are shown with the error. The unsent request is visible in local mode, so you can read the evidence before any model runs.
- Other options: keep `qwen2:0.5b` as the default (fast, and it will keep inventing); use `llama3.2:3b` or `qwen3:4b` (likely more careful, slower on CPU); a hosted key in the key field (same request body).

The key is kept in the browser's `localStorage` only. It is not written into the index. A saved key or model name from an earlier visit overrides the new default until you change the field.

Bump `EXTRACTOR_REV` in `pipeline.py` when extraction changes. Fingerprints include it, so the next Index re-reads files instead of skipping them forever.
