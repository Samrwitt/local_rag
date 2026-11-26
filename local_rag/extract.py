"""Turn one file into text plus interpreted time and place signals.

Text comes from the file itself when it has a text layer. Photos and
scanned PDFs go through Tesseract. OCR is interpreted into city, country,
language, and trip roles — not stored as raw OCR alone.

Videos are not opened frame-by-frame. Only container metadata (headers/tags)
is decoded into searchable trip data.
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
from local_rag.place import interpret_place

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm", ".rst", ".log"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

MAX_TEXT_CHARS = 200_000
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
SS_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass
class Extracted:
    kind: str
    text: str = ""
    captured_at: str | None = None
    time_source: str | None = None
    country: str | None = None
    city: str | None = None
    language: str | None = None
    roles: list[str] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)
    place_source: str | None = None
    lat: float | None = None
    lon: float | None = None
    warnings: list[str] = field(default_factory=list)
    ocr_used: bool = False
    interpreted: str = ""
    enrichment: dict = field(default_factory=dict)
    device: str | None = None
    duration: float | None = None
    content_hash: str | None = None


def kind_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext == ".pptx":
        return "pptx"
    if ext in {".xlsx", ".xlsm"}:
        return "xlsx"
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
        result = _extract_text(path)
    elif kind == "pdf":
        result = _extract_pdf(path)
    elif kind == "docx":
        result = _extract_docx(path)
    elif kind == "pptx":
        result = _extract_pptx(path)
    elif kind == "xlsx":
        result = _extract_xlsx(path)
    elif kind == "image":
        result = _extract_image(path)
    else:
        result = _extract_video(path)
    return _apply_place(result, path.name)


def _apply_place(result: Extracted, name: str) -> Extracted:
    place = interpret_place(
        result.text,
        path_name=name,
        gps_country=result.country,
    )
    if place.country:
        result.country = place.country
    result.city = place.city
    result.language = place.language
    result.roles = place.roles
    result.venues = place.venues
    result.place_source = place.place_source
    result.interpreted = "\n".join(place.summary_lines)
    result.enrichment = place.enrichment if hasattr(place, "enrichment") else {}
    return result


def _date_from_text(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"\b((?:19|20)\d{2})[-/.](\d{2})[-/.](\d{2})\b", text or "")
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            return datetime(year, month, day).isoformat(timespec="seconds"), "document"
        except ValueError:
            pass
    year_only = re.search(r"\b((?:19|20)\d{2})\b", text or "")
    if year_only:
        try:
            return datetime(int(year_only.group(1)), 1, 1).isoformat(timespec="seconds"), "document"
        except ValueError:
            return None, None
    return None, None


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
    body_date, body_source = _date_from_text(text)
    if body_date:
        captured, source = body_date, body_source
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
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
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
    captured, source = _date_from_text(text)
    if captured is None:
        captured, source = _time_from_mtime(path)
    return Extracted(
        kind="docx",
        text=text,
        captured_at=captured,
        time_source=source,
        warnings=warnings,
    )


def _extract_pptx(path: Path) -> Extracted:
    """Extract slide text from PPTX (OOXML) without external deps."""
    warnings = []
    slides: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                n for n in archive.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            )
            for name in names[:80]:
                root = ET.fromstring(archive.read(name))
                parts = [node.text or "" for node in root.iter(f"{A_NS}t") if node.text]
                line = " ".join(part.strip() for part in parts if part and part.strip())
                if line:
                    slides.append(f"Slide {len(slides) + 1}: {line}")
    except (zipfile.BadZipFile, KeyError, OSError, ET.ParseError) as exc:
        return Extracted(kind="pptx", warnings=[f"could not read pptx: {exc}"])
    text = "\n\n".join(slides)[:MAX_TEXT_CHARS]
    if not text:
        warnings.append("pptx had no text")
    captured, source = _date_from_text(text)
    if captured is None:
        captured, source = _time_from_mtime(path)
    return Extracted(
        kind="pptx",
        text=text,
        captured_at=captured,
        time_source=source,
        warnings=warnings,
    )


def _extract_xlsx(path: Path) -> Extracted:
    """Extract cell text from XLSX via sharedStrings + sheet XML."""
    warnings = []
    try:
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for si in root.iter(f"{SS_NS}si"):
                    texts = [t.text or "" for t in si.iter(f"{SS_NS}t")]
                    shared.append("".join(texts))
            rows_out: list[str] = []
            sheets = sorted(
                n for n in archive.namelist()
                if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            )
            for sheet_name in sheets[:10]:
                root = ET.fromstring(archive.read(sheet_name))
                for row in root.iter(f"{SS_NS}row"):
                    cells = []
                    for cell in row.iter(f"{SS_NS}c"):
                        ref = cell.attrib.get("r", "")
                        value_node = cell.find(f"{SS_NS}v")
                        if value_node is None or value_node.text is None:
                            continue
                        raw = value_node.text
                        if cell.attrib.get("t") == "s":
                            try:
                                raw = shared[int(raw)]
                            except (ValueError, IndexError):
                                pass
                        cells.append(f"{ref}:{raw}" if ref else raw)
                    if cells:
                        rows_out.append(" | ".join(cells))
                    if len(rows_out) >= 400:
                        break
                if len(rows_out) >= 400:
                    break
    except (zipfile.BadZipFile, KeyError, OSError, ET.ParseError) as exc:
        return Extracted(kind="xlsx", warnings=[f"could not read xlsx: {exc}"])
    text = "\n".join(rows_out)[:MAX_TEXT_CHARS]
    if not text:
        warnings.append("xlsx had no cells")
    captured, source = _date_from_text(text)
    if captured is None:
        captured, source = _time_from_mtime(path)
    return Extracted(
        kind="xlsx",
        text=text,
        captured_at=captured,
        time_source=source,
        warnings=warnings,
    )


def _extract_pdf(path: Path) -> Extracted:
    warnings = []
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
    captured, source = _date_from_text(text)
    if captured is None:
        captured, source = _pdf_creation_time(path)
    if captured is None:
        captured, source = _time_from_mtime(path)
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


def _open_image(path: Path):
    """Open common image formats; HEIC via pillow-heif or heif-convert fallback."""
    ext = path.suffix.lower()
    if ext in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener  # type: ignore

            register_heif_opener()
            return Image.open(path), None
        except Exception:
            pass
        with tempfile.TemporaryDirectory(prefix="localrag-heic-") as tmp:
            out = Path(tmp) / "converted.jpg"
            for cmd in (
                ["heif-convert", str(path), str(out)],
                ["magick", str(path), str(out)],
            ):
                try:
                    proc = subprocess.run(cmd, capture_output=True, timeout=45, check=False)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
                if out.exists() and out.stat().st_size > 0:
                    return Image.open(out), f"heic converted via {cmd[0]}"
            return None, "HEIC support unavailable (install pillow-heif or libheif tools)"
    return Image.open(path), None


def _extract_image(path: Path) -> Extracted:
    warnings = []
    lat = lon = None
    captured = source = None
    device = None
    try:
        lat, lon, captured, source, device = _image_exif(path)
    except Exception as exc:
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
        warnings.append("no GPS; place may still come from OCR text")
    if captured is None:
        captured, source = _time_from_name(path.name)
    if captured is None:
        captured, source = _time_from_mtime(path)

    ocr_path = path
    if path.suffix.lower() in {".heic", ".heif"}:
        image, heic_note = _open_image(path)
        if heic_note:
            warnings.append(heic_note)
        if image is None:
            return Extracted(
                kind="image",
                captured_at=captured,
                time_source=source,
                country=country,
                lat=lat,
                lon=lon,
                device=device,
                warnings=warnings,
            )
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.convert("RGB").save(tmp.name, format="JPEG", quality=90)
            ocr_path = Path(tmp.name)
        try:
            text, ocr_warning = _tesseract(ocr_path)
        finally:
            ocr_path.unlink(missing_ok=True)
    else:
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
        device=device,
        warnings=warnings,
        ocr_used=ocr_used,
    )


def _image_exif(path: Path) -> tuple[float | None, float | None, str | None, str | None, str | None]:
    image, note = _open_image(path)
    if image is None:
        raise RuntimeError(note or "unreadable image")
    exif = image.getexif()
    if not exif:
        return None, None, None, None, None
    captured = None
    source = None
    sub = exif.get_ifd(34665) if hasattr(exif, "get_ifd") else {}
    for key in (36867, 36868):
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
    make = exif.get(271)
    model = exif.get(272)
    device_parts = []
    for part in (make, model):
        if isinstance(part, bytes):
            part = part.decode("utf-8", "ignore")
        if part:
            device_parts.append(str(part).strip())
    device = " ".join(device_parts) or None
    return lat, lon, captured, source, device


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
    date_raw = gps.get(29)
    time_raw = gps.get(7)
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
    """Decode container metadata only. Do not open or OCR video frames."""
    warnings = []
    meta = _ffprobe(path)
    # A year in the file name (Italy_2019.mp4) beats a later container clock.
    captured, source = _time_from_name(path.name)
    if captured is None and meta.get("creation_time"):
        captured = _parse_iso_loose(meta["creation_time"])
        source = "container" if captured else None
    if captured is None:
        captured, source = _time_from_mtime(path)

    lines = [
        f"File: {path.name}",
        "Type: video",
        "Source: container metadata only (no frame reading, no speech).",
    ]
    if meta.get("duration") is not None:
        lines.append(f"Duration: {meta['duration']:.1f} seconds")
    if meta.get("format_name"):
        lines.append(f"Container: {meta['format_name']}")
    if meta.get("title"):
        lines.append(f"Title: {meta['title']}")
    if meta.get("comment"):
        lines.append(f"Comment: {meta['comment']}")
    if meta.get("location"):
        lines.append(f"Location tag: {meta['location']}")
    if meta.get("artist"):
        lines.append(f"Artist: {meta['artist']}")
    if meta.get("album"):
        lines.append(f"Album: {meta['album']}")
    if meta.get("description"):
        lines.append(f"Description: {meta['description']}")
    if meta.get("creation_time"):
        lines.append(f"Creation time tag: {meta['creation_time']}")
    # Path and tag words feed place interpretation (e.g. Italy_2019.mp4).
    lines.append(f"Path name: {path.name}")
    if not any(meta.get(key) for key in ("title", "comment", "location", "description")):
        warnings.append("video had few descriptive tags; place may come from the file name or date")

    country = None
    lat = lon = None
    if meta.get("lat") is not None and meta.get("lon") is not None:
        lat, lon = meta["lat"], meta["lon"]
        try:
            country = country_at(lat, lon)
        except FileNotFoundError as exc:
            warnings.append(str(exc))

    return Extracted(
        kind="video",
        text="\n".join(lines)[:MAX_TEXT_CHARS],
        captured_at=captured,
        time_source=source,
        country=country,
        lat=lat,
        lon=lon,
        duration=meta.get("duration"),
        device=meta.get("encoder") or meta.get("artist"),
        warnings=warnings,
        ocr_used=False,
    )


def _ffprobe(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:format_tags="
                "creation_time,title,comment,description,location,artist,album,encoder,com.apple.quicktime.location.ISO6709",
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
    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    duration = None
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None
    location = (
        tags.get("location")
        or tags.get("com.apple.quicktime.location.iso6709")
        or tags.get("location-eng")
    )
    lat = lon = None
    if location:
        match = re.search(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", str(location))
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
            except ValueError:
                lat = lon = None
    return {
        "duration": duration,
        "format_name": fmt.get("format_name"),
        "creation_time": tags.get("creation_time"),
        "title": tags.get("title"),
        "comment": tags.get("comment"),
        "description": tags.get("description"),
        "location": location,
        "artist": tags.get("artist"),
        "album": tags.get("album"),
        "encoder": tags.get("encoder"),
        "lat": lat,
        "lon": lon,
    }


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
    cleaned = re.sub(r"[^\S\n]+", " ", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    words = re.findall(r"[A-Za-z]{3,}", cleaned)
    if len(words) < 5:
        return None
    letters = re.findall(r"[A-Za-z]", cleaned)
    if not letters or len(words) * 4 < len(letters) / 2:
        return None
    return cleaned[:MAX_TEXT_CHARS]


def _time_from_mtime(path: Path) -> tuple[str, str]:
    stamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return stamp, "file_modified"


def _time_from_name(name: str) -> tuple[str | None, str | None]:
    match = re.search(r"(20\d{2}|19\d{2})[-_.](\d{2})[-_.](\d{2})", name)
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            return datetime(year, month, day).isoformat(timespec="seconds"), "filename"
        except ValueError:
            pass
    # Prefer a year in the file name over a later container rewrite when present.
    year_only = re.search(r"(?:^|[_\W])((?:19|20)\d{2})(?:[_\W]|$)", name)
    if year_only:
        try:
            return datetime(int(year_only.group(1)), 1, 1).isoformat(timespec="seconds"), "filename"
        except ValueError:
            return None, None
    return None, None


def _parse_iso_loose(raw: str) -> str | None:
    raw = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=None).isoformat(timespec="seconds")
    except ValueError:
        return None
