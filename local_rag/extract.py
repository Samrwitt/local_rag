"""Turn one file into plain text plus time and place signals.

Text comes from the file itself when it has a text layer. Photos and
videos go through Tesseract (English). Country comes only from GPS,
never from words in the text.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from local_rag.geo import country_at

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm", ".rst", ".log"}
DOC_EXT = {".pdf", ".docx"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

MAX_TEXT_CHARS = 200_000
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class Extracted:
    kind: str
    text: str = ""
    captured_at: str | None = None
    time_source: str | None = None
    country: str | None = None
    lat: float | None = None
    lon: float | None = None
    warnings: list[str] = field(default_factory=list)
    ocr_used: bool = False


def kind_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    return None


def extract(path: Path) -> Extracted:
    kind = kind_for(path)
    if kind is None:
        return Extracted(kind="skipped", warnings=["unsupported type"])
    if kind == "text":
        return _extract_text(path)
    if kind == "pdf":
        return _extract_pdf(path)
    if kind == "docx":
        return _extract_docx(path)
    if kind == "image":
        return _extract_image(path)
    return _extract_video(path)


def _extract_text(path: Path) -> Extracted:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
        raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = raw[:MAX_TEXT_CHARS]
    warnings = []
    if len(raw) > MAX_TEXT_CHARS:
        warnings.append(f"text truncated to {MAX_TEXT_CHARS} characters")
    captured, source = _time_from_name(path.name)
    if captured is None:
        captured, source = _time_from_mtime(path)
    return Extracted(
        kind="text",
        text=text,
        captured_at=captured,
        time_source=source,
        warnings=warnings,
    )


def _extract_docx(path: Path) -> Extracted:
    warnings = []
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        return Extracted(kind="docx", warnings=[f"could not read docx: {exc}"])
    root = ET.fromstring(xml)
    paragraphs = []
    for para in root.iter(f"{W_NS}p"):
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{W_NS}t":
                parts.append(node.text or "")
            elif node.tag == f"{W_NS}tab":
                parts.append("\t")
            elif node.tag == f"{W_NS}br":
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    text = "\n\n".join(paragraphs)[:MAX_TEXT_CHARS]
    if not text:
        warnings.append("docx had no text")
    captured, source = _time_from_mtime(path)
    return Extracted(
        kind="docx",
        text=text,
        captured_at=captured,
        time_source=source,
        warnings=warnings,
    )


def _extract_pdf(path: Path) -> Extracted:
    warnings = []
    captured, source = _pdf_creation_time(path)
    if captured is None:
        captured, source = _time_from_mtime(path)
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        proc = None
        warnings.append(f"pdftotext failed: {exc}")
    text = (proc.stdout if proc else "") or ""
    text = text.strip()
    ocr_used = False
    # A scanned page returns almost nothing. Rasterize a few pages and OCR them.
    if len(re.sub(r"\s+", "", text)) < 40:
        ocr_text, ocr_warning = _ocr_pdf_pages(path)
        if ocr_warning:
            warnings.append(ocr_warning)
        if ocr_text:
            text = ocr_text
            ocr_used = True
            warnings.append("pdf had no text layer; used OCR")
        elif not text:
            warnings.append("pdf produced no text")
    return Extracted(
        kind="pdf",
        text=text[:MAX_TEXT_CHARS],
        captured_at=captured,
        time_source=source,
        warnings=warnings,
        ocr_used=ocr_used,
    )


def _pdf_creation_time(path: Path) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            ["pdfinfo", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, None
    for line in proc.stdout.splitlines():
        if line.startswith("CreationDate:"):
            raw = line.split(":", 1)[1].strip()
            parsed = _parse_pdf_date(raw)
            if parsed:
                return parsed, "pdf"
    return None, None


def _parse_pdf_date(raw: str) -> str | None:
    # pdfinfo: "Sun Nov 23 12:27:08 2025 EAT"
    head = raw[:24].strip()
    for fmt in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(head, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    match = re.search(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", raw)
    if match:
        y, mo, d, h, mi, s = match.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s)).isoformat(timespec="seconds")
        except ValueError:
            return None
    return None


def _ocr_pdf_pages(path: Path, max_pages: int = 3) -> tuple[str, str | None]:
    pages = _pdf_page_count(path)
    last = min(pages or 1, max_pages)
    with tempfile.TemporaryDirectory(prefix="localrag-pdf-") as tmp:
        prefix = str(Path(tmp) / "page")
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "140", "-f", "1", "-l", str(last), str(path), prefix],
                capture_output=True,
                timeout=90,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return "", f"could not rasterize pdf: {exc}"
        chunks = []
        for image in sorted(Path(tmp).glob("page*.png")):
            text, warning = _tesseract(image)
            if warning:
                return "", warning
            kept = _usable_ocr(text)
            if kept:
                chunks.append(kept)
        if not chunks:
            return "", "OCR on the pdf pages did not produce readable words"
        if pages and pages > max_pages:
            return "\n\n".join(chunks), f"OCR stopped after {max_pages} of {pages} pages"
        return "\n\n".join(chunks), None


def _pdf_page_count(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["pdfinfo", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _extract_image(path: Path) -> Extracted:
    warnings = []
    lat = lon = None
    captured = source = None
    try:
        lat, lon, captured, source = _image_exif(path)
    except Exception as exc:  # a bad image should not stop the crawl
        warnings.append(f"exif unreadable: {exc}")
    country = None
    if lat is not None and lon is not None:
        try:
            country = country_at(lat, lon)
        except FileNotFoundError as exc:
            warnings.append(str(exc))
        if country is None:
            warnings.append("GPS point did not fall inside a country polygon")
    else:
        warnings.append("no GPS latitude/longitude (country left blank)")
    if captured is None:
        captured, source = _time_from_name(path.name)
    if captured is None:
        captured, source = _time_from_mtime(path)

    text, ocr_warning = _tesseract(path)
    ocr_used = False
    kept = _usable_ocr(text) if text else None
    if ocr_warning:
        warnings.append(ocr_warning)
    elif not kept:
        warnings.append("OCR discarded: not enough readable words")
        text = ""
    else:
        text = kept
        ocr_used = True
    return Extracted(
        kind="image",
        text=text,
        captured_at=captured,
        time_source=source,
        country=country,
        lat=lat,
        lon=lon,
        warnings=warnings,
        ocr_used=ocr_used,
    )


def _image_exif(path: Path) -> tuple[float | None, float | None, str | None, str | None]:
    image = Image.open(path)
    exif = image.getexif()
    if not exif:
        return None, None, None, None
    captured = None
    source = None
    sub = exif.get_ifd(34665) if hasattr(exif, "get_ifd") else {}
    for key in (36867, 36868):  # DateTimeOriginal, DateTimeDigitized
        raw = sub.get(key) if sub else None
        parsed = _parse_exif_datetime(raw)
        if parsed:
            captured, source = parsed, "exif"
            break
    if captured is None:
        parsed = _parse_exif_datetime(exif.get(306))
        if parsed:
            captured, source = parsed, "exif"
    captured = _with_offset(captured, sub)
    gps = exif.get_ifd(34853) if hasattr(exif, "get_ifd") else {}
    lat, lon = _gps_lat_lon(gps)
    if captured is None and gps:
        parsed = _gps_datetime(gps)
        if parsed:
            captured, source = parsed, "gps"
    return lat, lon, captured, source


def _gps_lat_lon(gps: dict) -> tuple[float | None, float | None]:
    if not gps or 2 not in gps or 4 not in gps:
        return None, None
    try:
        lat = _dms_to_deg(gps[2])
        lon = _dms_to_deg(gps[4])
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None
    lat_ref = gps.get(1, "N")
    lon_ref = gps.get(3, "E")
    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode("ascii", "ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode("ascii", "ignore")
    if str(lat_ref).upper().startswith("S"):
        lat = -lat
    if str(lon_ref).upper().startswith("W"):
        lon = -lon
    return lat, lon


def _ratio(value) -> float:
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    return float(value)


def _dms_to_deg(value) -> float:
    degrees, minutes, seconds = value
    return _ratio(degrees) + _ratio(minutes) / 60 + _ratio(seconds) / 3600


def _with_offset(captured: str | None, sub: dict | None) -> str | None:
    """Keep the camera's timezone when EXIF recorded one.

    DateTimeOriginal is local camera time. OffsetTimeOriginal says how far
    that clock was from UTC. GPS time is usually UTC, so the two disagree
    by exactly that offset when both exist.
    """
    if not captured or not sub:
        return captured
    offset = sub.get(36881) or sub.get(36882) or sub.get(36880)
    if isinstance(offset, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", offset.strip()):
        return captured + offset.strip()
    return captured


def _parse_exif_datetime(raw) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).isoformat(timespec="seconds")
        except ValueError:
            continue
    return None


def _gps_datetime(gps: dict) -> str | None:
    date_raw = gps.get(29)  # GPSDateStamp
    time_raw = gps.get(7)  # GPSTimeStamp
    if not isinstance(date_raw, str):
        return None
    try:
        year, month, day = [int(part) for part in date_raw.split(":")]
        if time_raw and len(time_raw) >= 3:
            hour, minute, second = (_ratio(time_raw[0]), _ratio(time_raw[1]), _ratio(time_raw[2]))
            return datetime(year, month, day, int(hour), int(minute), int(second)).isoformat(timespec="seconds")
        return datetime(year, month, day).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _extract_video(path: Path) -> Extracted:
    warnings = []
    meta = _ffprobe(path)
    captured = None
    source = None
    if meta.get("creation_time"):
        captured = _parse_iso_loose(meta["creation_time"])
        source = "container" if captured else None
    if captured is None:
        captured, source = _time_from_name(path.name)
    if captured is None:
        captured, source = _time_from_mtime(path)
    duration = meta.get("duration")
    ocr_parts = []
    frame_notes = []
    with tempfile.TemporaryDirectory(prefix="localrag-video-") as tmp:
        timestamps = _frame_times(duration)
        for index, seconds in enumerate(timestamps):
            frame = Path(tmp) / f"frame-{index}.jpg"
            ok, err = _grab_frame(path, seconds, frame)
            if not ok:
                frame_notes.append(err or f"frame at {seconds:.1f}s failed")
                continue
            text, ocr_warning = _tesseract(frame)
            if ocr_warning:
                warnings.append(ocr_warning)
                break
            kept = _usable_ocr(text)
            if kept:
                ocr_parts.append(f"[frame {seconds:.1f}s]\n{kept}")
    if not ocr_parts:
        warnings.append("no readable on-screen text in sampled frames")
    if frame_notes and not ocr_parts:
        warnings.append(frame_notes[0])
    header = []
    if duration:
        header.append(f"Duration: {duration:.1f} seconds.")
    header.append("Speech was not transcribed. Only sampled frames were read.")
    text = "\n".join(header)
    if ocr_parts:
        text = text + "\n\n" + "\n\n".join(ocr_parts)
    return Extracted(
        kind="video",
        text=text[:MAX_TEXT_CHARS],
        captured_at=captured,
        time_source=source,
        warnings=warnings,
        ocr_used=bool(ocr_parts),
    )


def _ffprobe(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:format_tags=creation_time",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    fmt = payload.get("format") or {}
    tags = fmt.get("tags") or {}
    duration = None
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None
    return {"duration": duration, "creation_time": tags.get("creation_time")}


def _frame_times(duration: float | None) -> list[float]:
    if not duration or duration < 0.5:
        return [0.0]
    if duration < 3:
        return [duration / 2]
    return [duration * 0.15, duration * 0.5, duration * 0.85]


def _grab_frame(path: Path, seconds: float, dest: Path) -> tuple[bool, str | None]:
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seconds:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                "scale=1280:-1",
                str(dest),
            ],
            capture_output=True,
            timeout=40,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, str(exc)
    if proc.returncode != 0 or not dest.exists():
        tail = (proc.stderr or b"")[-200:].decode("utf-8", "ignore")
        return False, tail.strip() or "ffmpeg produced no frame"
    return True, None


def _tesseract(path: Path) -> tuple[str, str | None]:
    try:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", "OCR timed out"
    except FileNotFoundError:
        return "", "tesseract is not installed"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return "", f"OCR failed: {err[-1] if err else 'unknown'}"
    return proc.stdout or "", None


def _usable_ocr(text: str) -> str | None:
    """Keep OCR only when it contains real words, not chart noise."""
    cleaned = re.sub(r"[^\S\n]+", " ", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    words = re.findall(r"[A-Za-z]{3,}", cleaned)
    if len(words) < 5:
        return None
    # Mostly single-character junk still produces a few accidental words.
    letters = re.findall(r"[A-Za-z]", cleaned)
    if not letters or len(words) * 4 < len(letters) / 2:
        return None
    return cleaned[:MAX_TEXT_CHARS]


def _time_from_mtime(path: Path) -> tuple[str, str]:
    stamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return stamp, "file_modified"


def _time_from_name(name: str) -> tuple[str | None, str | None]:
    match = re.search(r"(20\d{2})[-_.](\d{2})[-_.](\d{2})", name)
    if not match:
        return None, None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day).isoformat(timespec="seconds"), "filename"
    except ValueError:
        return None, None


def _parse_iso_loose(raw: str) -> str | None:
    raw = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=None).isoformat(timespec="seconds")
    except ValueError:
        return None
