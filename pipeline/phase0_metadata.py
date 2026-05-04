"""Phase 0 — EXIF extraction and OCR."""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pytesseract
import numpy as np


def _convert_gps_to_decimal(gps_coord, gps_ref):
    """Convert GPS coordinates from EXIF DMS format to decimal degrees."""
    try:
        degrees = float(gps_coord[0])
        minutes = float(gps_coord[1])
        seconds = float(gps_coord[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        if gps_ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, IndexError, ZeroDivisionError):
        return None


def extract_exif(image: Image.Image) -> dict | None:
    """Extract EXIF metadata from a PIL Image. Returns None if no EXIF data."""
    try:
        exif_data = image._getexif()
    except (AttributeError, Exception):
        return None

    if not exif_data:
        return None

    parsed = {}
    gps_info = {}

    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, str(tag_id))

        if tag_name == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                gps_info[gps_tag_name] = gps_value
        elif tag_name in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
            parsed["timestamp"] = str(value)
        elif tag_name == "Make":
            parsed["camera_make"] = str(value)
        elif tag_name == "Model":
            parsed["camera_model"] = str(value)

    # Parse GPS coordinates
    if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
        lat = _convert_gps_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = _convert_gps_to_decimal(
            gps_info.get("GPSLongitude", []),
            gps_info.get("GPSLongitudeRef", ""),
        )
        if lat is not None and lon is not None:
            parsed["gps_lat"] = lat
            parsed["gps_lon"] = lon

    return parsed if parsed else None


_OCR_BLOCKLIST = {"l", "i", "|", "1", "ll", "ii", "ii'", "i'", "o", "0"}


def _clean(text: str | None) -> str | None:
    """Reject low-quality OCR strings (single chars, gibberish, blocklisted)."""
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    lowered = t.lower()
    if lowered in _OCR_BLOCKLIST:
        return None
    if lowered in {"none", "n/a", "nothing", "no text", "no text visible", "<end>"}:
        return None
    alnum = sum(c.isalnum() for c in t)
    if alnum < 2:
        return None
    return t


def _is_gibberish(text: str) -> bool:
    """Reject text without at least one word of 3+ letters."""
    words = text.split()
    has_real_word = any(
        sum(c.isalpha() for c in w) >= 3 for w in words
    )
    return not has_real_word


def extract_ocr_text(image: Image.Image) -> str | None:
    """Run OCR on a PIL Image. Tries English, Russian, and Ukrainian. Returns None if no usable text."""
    best = None
    for lang in ("eng+rus+ukr", "eng", "rus+ukr"):
        try:
            text = pytesseract.image_to_string(image, lang=lang).strip()
            cleaned = _clean(text)
            if cleaned and not _is_gibberish(cleaned):
                if best is None or len(cleaned) > len(best):
                    best = cleaned
        except Exception:
            continue
    return best


def run_phase0(image: Image.Image) -> dict:
    """Run full Phase 0: EXIF + OCR. Returns structured result dict."""
    return {
        "exif": extract_exif(image),
        "ocr_text": extract_ocr_text(image),
    }
