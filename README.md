# Local RAG

A folder on this computer goes in. Passages that match a question come out. A language model is optional.

The previous attempt at this was a multi-service platform (Postgres, Redis, Qdrant, MinIO, Celery, hosted API keys). It could not be opened on a real folder without that stack. This project is one Python process and one SQLite file.

## Path

1. **Crawl** a folder for text, PDF, DOCX, photos, and video.
2. **Clean and chunk.** Pull text, OCR where there is no text layer, and record time and country.
3. **Vectorize** with the MiniLM model already cached on this machine.
4. **Ask in the browser.** Local preview shows the passages immediately.
5. **Call a model** only if you turn that on. The request is the OpenAI chat-completions shape, so Ollama and most hosted APIs both fit.

## Run

```bash
cd /home/lenovo/local_rag
python3 -m local_rag serve
```

Open http://127.0.0.1:8765 and enter a folder. Do not point it at `/` or your whole home directory.

Command line, if you do not want the browser:

```bash
python3 -m local_rag index /path/to/folder
python3 -m local_rag ask "your question"
```

The index is `data/index.sqlite` (gitignored).

## What each file contributes

| Signal | Where it comes from | Where it does not come from |
| --- | --- | --- |
| Text | `.txt` `.md` `.csv` `.json` `.html`, PDF text layer, DOCX | Guessing from a filename |
| OCR | Photos, video frames, PDFs with no text layer. English Tesseract only. Chart junk is dropped. | Speech in a video |
| Time | EXIF, GPS timestamp, PDF creation date, video container, a date in the filename, else file modified time | The clock of the machine that indexed it, unless nothing else exists |
| Country | GPS latitude/longitude inside an offline country polygon | Words in the document. A note about Ethiopia is not a photo taken in Ethiopia. |

Every file also gets a short metadata card (name, type, time, country) so a photo with no readable text can still be found by place and date.

## Language model API

Local preview does not call anything. The UI still shows the JSON that would be sent.

```http
POST {base}/chat/completions
Content-Type: application/json
Authorization: Bearer <key>     # omitted when the key is empty
```

```json
{
  "model": "qwen2.5:1.5b",
  "temperature": 0.2,
  "messages": [
    {"role": "system", "content": "Use only the numbered sources..."},
    {"role": "user", "content": "Question: ...\n\nSources:\n[1] file | kind | time | country\n..."}
  ]
}
```

Ollama's default base is `http://127.0.0.1:11434/v1`. The default model is `qwen2.5:1.5b`. A hosted provider uses the same body with its own base URL, model name, and key. The key stays in the browser (`localStorage`) and is not written into the index.

If the call fails, the answer falls back to the same passages the local preview would have shown.

Our own HTTP API, all on `127.0.0.1`:

| Method | Path | Body |
| --- | --- | --- |
| POST | `/api/index` | `{"path": "/folder", "max_files": 400}` |
| GET | `/api/job` | index progress |
| GET | `/api/documents` | what was extracted |
| POST | `/api/query` | `{"question", "mode": "local" or "llm", "kind", "country", "year", "llm_base_url", "llm_model", "llm_api_key"}` |

## Trial folder

`corpus/` is gitignored. It is a handful of real files from this machine (notes, a PDF invoice, a DOCX, a GPS photo, a short video, a screenshot saved as a PDF) used to try the pipeline as a person would. Point the UI at that folder, or at any other folder of yours.

Obstacles that showed up while doing that, and the options for each, are in [PROBLEMS.md](PROBLEMS.md).
