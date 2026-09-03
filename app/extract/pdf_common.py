from __future__ import annotations

import io
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import fitz
import pdfplumber

from app.storage.blobs import blob_store

HEADER_Y_MAX = 0.08
FOOTER_Y_MIN = 0.92
COLUMN_GAP_RATIO = 0.18


@dataclass
class TextBlock:
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    bold: bool
    kind: str = "body"

    @property
    def bbox(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class PageProfile:
    page_number: int
    width: float
    height: float
    char_count: int
    word_count: int
    image_coverage: float
    text_density: float
    table_count: int
    image_count: int
    font_sizes: list[float]
    label: str
    method: str = "pymupdf"
    warnings: list[str] = field(default_factory=list)


def open_pdf(path: str | bytes) -> fitz.Document:
    if isinstance(path, bytes):
        return fitz.open(stream=path, filetype="pdf")
    return fitz.open(path)


def pdf_metadata(doc: fitz.Document) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "author": meta.get("author") or "",
        "creator": meta.get("creator") or "",
        "producer": meta.get("producer") or "",
        "created": meta.get("creationDate") or "",
        "modified": meta.get("modDate") or "",
        "keywords": meta.get("keywords") or "",
        "subject": meta.get("subject") or "",
        "title": meta.get("title") or "",
    }


def _block_font(page: fitz.Page, bbox: fitz.Rect) -> tuple[float, str, bool]:
    sizes: list[float] = []
    names: list[str] = []
    bold = False
    for span in page.get_text("dict")["blocks"]:
        if span.get("type") != 0:
            continue
        for line in span.get("lines", []):
            for s in line.get("spans", []):
                r = fitz.Rect(s["bbox"])
                if r.intersects(bbox):
                    sizes.append(float(s.get("size") or 0))
                    names.append(s.get("font") or "")
                    if "bold" in (s.get("font") or "").lower() or (s.get("flags", 0) & 16):
                        bold = True
    if not sizes:
        return 0.0, "", False
    return sum(sizes) / len(sizes), Counter(names).most_common(1)[0][0], bold


def extract_blocks(doc: fitz.Document) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("blocks")
        for b in raw:
            x0, y0, x1, y1, text, *_ = b[:6] if len(b) >= 5 else (*b, "")
            text = (text or "").strip()
            if not text:
                continue
            bbox = fitz.Rect(x0, y0, x1, y1)
            size, font, bold = _block_font(page, bbox)
            blocks.append(
                TextBlock(
                    page=i,
                    text=re.sub(r"[ \t]+", " ", text),
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    font_size=size,
                    font_name=font,
                    bold=bold,
                )
            )
    return blocks


def cluster_columns(blocks: list[TextBlock], page_width: float) -> list[TextBlock]:
    if not blocks:
        return blocks
    xs = sorted(b.x0 for b in blocks)
    if len(xs) < 4:
        return sorted(blocks, key=lambda b: (b.y0, b.x0))
    gap = page_width * COLUMN_GAP_RATIO
    cuts = [0]
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > gap:
            cuts.append(i)
    if len(cuts) < 2:
        return sorted(blocks, key=lambda b: (b.y0, b.x0))
    # assign column index by x0 cluster
    centers = []
    cuts.append(len(xs))
    for a, b in zip(cuts, cuts[1:]):
        centers.append(sum(xs[a:b]) / max(1, b - a))
    centers = sorted(set(round(c, 1) for c in centers))

    def col(block: TextBlock) -> int:
        return min(range(len(centers)), key=lambda i: abs(block.x0 - centers[i]))

    return sorted(blocks, key=lambda b: (col(b), b.y0, b.x0))


def reorder_page_blocks(blocks: list[TextBlock], widths: dict[int, float]) -> list[TextBlock]:
    by_page: dict[int, list[TextBlock]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)
    ordered: list[TextBlock] = []
    for page in sorted(by_page):
        ordered.extend(cluster_columns(by_page[page], widths.get(page, 612)))
    return ordered


def detect_boilerplate(blocks: list[TextBlock], page_heights: dict[int, float], page_count: int) -> set[str]:
    chrome = {"confidential draft", "studyfetch seed corpus", "headers repeat"}
    extras = {k for k in chrome}
    if page_count < 2:
        for b in blocks:
            key = re.sub(r"\d+", "#", b.text.strip().lower())
            if any(c in key for c in chrome) or re.search(r"\bpage\s+#\b", key):
                extras.add(key)
        return extras
    candidates: Counter[str] = Counter()
    for b in blocks:
        h = page_heights.get(b.page, 792)
        rel_top = b.y0 / h
        rel_bot = b.y1 / h
        if rel_top <= HEADER_Y_MAX or rel_bot >= FOOTER_Y_MIN:
            key = re.sub(r"\d+", "#", b.text.strip().lower())
            if 2 <= len(key) <= 80:
                candidates[key] += 1
    threshold = max(2, int(page_count * 0.3))
    junk = {k for k, n in candidates.items() if n >= threshold}
    for b in blocks:
        key = re.sub(r"\d+", "#", b.text.strip().lower())
        if any(c in key for c in chrome):
            junk.add(key)
    return junk


def strip_boilerplate(blocks: list[TextBlock], junk: set[str], page_heights: dict[int, float]) -> tuple[list[TextBlock], list[TextBlock]]:
    kept: list[TextBlock] = []
    removed: list[TextBlock] = []
    for b in blocks:
        h = page_heights.get(b.page, 792)
        rel_top = b.y0 / h
        rel_bot = b.y1 / h
        key = re.sub(r"\d+", "#", b.text.strip().lower())
        if key in junk and (rel_top <= HEADER_Y_MAX or rel_bot >= FOOTER_Y_MIN):
            b.kind = "boilerplate"
            removed.append(b)
        else:
            kept.append(b)
    return kept, removed


def heading_level(block: TextBlock, body_size: float) -> int | None:
    text = block.text.strip()
    if not text or len(text) > 160:
        return None
    if text.count("\n") > 3:
        return None
    numbered = bool(re.match(r"^(\d+(\.\d+){0,3}|[A-Z]\.|(Chapter|Section|Appendix)\b)", text))
    size_delta = block.font_size - body_size
    if block.font_size <= 0:
        return 1 if numbered and len(text) < 80 else None
    if size_delta >= 6 or (block.bold and size_delta >= 2.5) or (numbered and size_delta >= 0.5):
        if size_delta >= 8 or re.match(r"^(Chapter|Appendix)\b", text, re.I):
            return 1
        if re.match(r"^\d+\.\d+", text) or size_delta < 5:
            return 2 if size_delta >= 2 else 3
        return 1
    return None


def infer_body_size(blocks: list[TextBlock]) -> float:
    sizes = [round(b.font_size, 1) for b in blocks if b.font_size > 0 and len(b.text) > 40]
    if not sizes:
        return 11.0
    return Counter(sizes).most_common(1)[0][0]


def build_sections(blocks: list[TextBlock], title: str) -> list[dict[str, Any]]:
    body_size = infer_body_size(blocks)
    sections: list[dict[str, Any]] = []
    current = {
        "id": f"sec_{uuid.uuid4().hex[:10]}",
        "parent_id": None,
        "level": 1,
        "title": title or "Document",
        "page_start": blocks[0].page if blocks else 1,
        "page_end": blocks[0].page if blocks else 1,
        "blocks": [],
    }
    stack: list[dict[str, Any]] = [current]
    sections.append(current)

    for b in blocks:
        level = heading_level(b, body_size)
        if level is None:
            stack[-1]["blocks"].append(b)
            stack[-1]["page_end"] = b.page
            continue
        b.kind = "heading"
        heading = re.sub(r"\s+", " ", b.text).strip()
        if current is stack[0] and re.sub(r"[\W_]+", "", heading.lower()) == re.sub(
            r"[\W_]+", "", (current["title"] or "").lower()
        ):
            current["page_start"] = min(current["page_start"], b.page)
            current["page_end"] = max(current["page_end"], b.page)
            continue
        node = {
            "id": f"sec_{uuid.uuid4().hex[:10]}",
            "parent_id": None,
            "level": level,
            "title": heading,
            "page_start": b.page,
            "page_end": b.page,
            "blocks": [],
        }
        while stack and stack[-1]["level"] >= level:
            if len(stack) == 1:
                break
            stack.pop()
        parent = stack[-1] if stack else None
        node["parent_id"] = parent["id"] if parent else None
        stack.append(node)
        sections.append(node)
    return sections


def section_text(section: dict[str, Any]) -> str:
    return "\n".join(b.text for b in section.get("blocks", []) if b.kind != "heading")


def recursive_split(text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    seps = ["\n\n", "\n", ". ", " "]
    parts: list[str] = []

    def split_once(chunk: str, sep_i: int) -> list[str]:
        if len(chunk) <= max_chars or sep_i >= len(seps):
            return [chunk] if chunk.strip() else []
        sep = seps[sep_i]
        pieces = chunk.split(sep)
        buf = ""
        out: list[str] = []
        for p in pieces:
            cand = p if not buf else buf + sep + p
            if len(cand) <= max_chars:
                buf = cand
            else:
                if buf:
                    out.extend(split_once(buf, sep_i + 1))
                buf = p
        if buf:
            out.extend(split_once(buf, sep_i + 1))
        return out

    raw = split_once(text, 0)
    if overlap <= 0 or len(raw) <= 1:
        return raw
    merged: list[str] = []
    for i, part in enumerate(raw):
        if i == 0:
            merged.append(part)
            continue
        prev_tail = merged[-1][-overlap:]
        merged.append((prev_tail + " " + part).strip())
    return merged


def page_profiles(doc: fitz.Document, plumber_path: str | None = None) -> list[PageProfile]:
    tables_by_page: dict[int, int] = defaultdict(int)
    if plumber_path:
        with pdfplumber.open(plumber_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    tables_by_page[i] = len(page.find_tables() or [])
                except Exception:
                    tables_by_page[i] = 0

    profiles: list[PageProfile] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        words = [w for w in re.findall(r"[A-Za-z0-9]{2,}", text)]
        images = page.get_images(full=True)
        img_area = 0.0
        page_area = max(page.rect.width * page.rect.height, 1)
        for img in page.get_image_info():
            bbox = img.get("bbox")
            if bbox:
                img_area += max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        coverage = min(1.0, img_area / page_area)
        density = len(text) / page_area * 1000
        sizes = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(float(span.get("size") or 0))
        label = classify_page(len(text), coverage, tables_by_page[i], density)
        profiles.append(
            PageProfile(
                page_number=i,
                width=page.rect.width,
                height=page.rect.height,
                char_count=len(text),
                word_count=len(words),
                image_coverage=round(coverage, 4),
                text_density=round(density, 4),
                table_count=tables_by_page[i],
                image_count=len(images),
                font_sizes=sizes[:40],
                label=label,
            )
        )
    return profiles


def classify_page(char_count: int, image_coverage: float, table_count: int, density: float) -> str:
    if char_count < 80 and image_coverage > 0.55:
        return "scanned"
    if char_count < 80:
        return "low_text"
    if table_count >= 1 and char_count < 2500:
        return "table_heavy"
    if image_coverage > 0.28:
        return "figure_heavy"
    if density > 0:
        return "digital_text"
    return "mixed"


def extract_images(doc: fitz.Document, document_id: str) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for i, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                data = pix.tobytes("png")
            except Exception:
                continue
            if len(data) < 800:
                continue
            key = f"{document_id}/p{i}_{xref}.png"
            uri = blob_store.put(key, data, "image/png")
            assets.append(
                {
                    "id": f"fig_{uuid.uuid4().hex[:10]}",
                    "document_id": document_id,
                    "page_number": i,
                    "asset_type": "figure",
                    "caption": "",
                    "blob_uri": uri,
                    "bbox_json": "[]",
                    "extra_json": "{}",
                }
            )
    return assets


def attach_captions(assets: list[dict[str, Any]], blocks: list[TextBlock]) -> None:
    caption_re = re.compile(r"^(figure|fig\.|table|diagram|chart)\s*[\d.]*", re.I)
    by_page: dict[int, list[TextBlock]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)
    for asset in assets:
        page = asset["page_number"]
        kind = (asset.get("asset_type") or "").lower()
        picked = ""
        for b in by_page.get(page, []):
            line = (b.text or "").strip().splitlines()[0].strip() if (b.text or "").strip() else ""
            match = caption_re.match(line)
            if not match:
                continue
            label = match.group(1).lower()
            if kind == "figure" and label == "table":
                continue
            if kind == "table" and label in {"figure", "fig."}:
                continue
            picked = line[:240]
            break
        if picked:
            asset["caption"] = picked


def extract_tables_plumber(path: str, document_id: str) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                headers = [str(c or "").strip() for c in table[0]]
                rows = [[str(c or "").strip() for c in row] for row in table[1:]]
                md = _table_markdown(headers, rows)
                assets.append(
                    {
                        "id": f"tbl_{uuid.uuid4().hex[:10]}",
                        "document_id": document_id,
                        "page_number": i,
                        "asset_type": "table",
                        "caption": "",
                        "blob_uri": "",
                        "bbox_json": "[]",
                        "extra_json": __import__("json").dumps(
                            {"headers": headers, "rows": rows, "markdown": md, "index": idx}
                        ),
                    }
                )
    return assets


def extract_tables_pymupdf(doc: fitz.Document, document_id: str) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for i, page in enumerate(doc, start=1):
        try:
            finder = page.find_tables()
            tables = finder.tables if finder else []
        except Exception:
            tables = []
        for idx, table in enumerate(tables):
            data = table.extract()
            if not data or len(data) < 2:
                continue
            headers = [str(c or "").strip() for c in data[0]]
            rows = [[str(c or "").strip() for c in row] for row in data[1:]]
            md = _table_markdown(headers, rows)
            assets.append(
                {
                    "id": f"tbl_{uuid.uuid4().hex[:10]}",
                    "document_id": document_id,
                    "page_number": i,
                    "asset_type": "table",
                    "caption": "",
                    "blob_uri": "",
                    "bbox_json": str(list(table.bbox)) if getattr(table, "bbox", None) else "[]",
                    "extra_json": __import__("json").dumps(
                        {"headers": headers, "rows": rows, "markdown": md, "index": idx}
                    ),
                }
            )
    return assets


def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r[: len(headers)]) + " |" for r in rows)
    return "\n".join([line, sep, body])


def ocr_page(page: fitz.Page) -> tuple[str, str]:
    """OCR a page. Returns (text, method). Empty text if OCR is unavailable."""
    from app.usage import add as usage_add

    usage_add("ocr_pages", 1)
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        usage_add("ocr_failed", 1)
        return "", "ocr_unavailable"
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        text = pytesseract.image_to_string(img) or ""
    except Exception:
        usage_add("ocr_failed", 1)
        return "", "ocr_failed"
    return text.strip(), "tesseract"


def page_to_png(page: fitz.Page) -> bytes:
    return page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")


def as_dicts(blocks: list[TextBlock]) -> list[dict[str, Any]]:
    return [asdict(b) for b in blocks]


NUMBER_RE = re.compile(
    r"(?P<name>(?:Maximum|Nominal|Peak|Average|Min|Max|Typical|Q[1-4])?"
    r"[ \t]*[A-Z][A-Za-z][A-Za-z0-9 /-]{1,40}):[ \t]*"
    r"(?P<raw>\$?-?\d[\d,]*(?:\.\d+)?)"
    r"(?:[ \t]+(?P<unit>V|A|W|kW|MHz|GHz|USD|ms|mm|kg|C|°C|%))?(?=\s|$|[.,;:])"
)
BARE_METRIC_RE = re.compile(
    r"(?<![\w-])(?P<raw>\$?-?\d[\d,]*(?:\.\d+)?)\s+(?P<unit>V|A|W|kW|MHz|GHz|USD|ms|mm|kg|%)(?=\s|$|[.,;:])"
)
# Min / typ / max rows: "Operating Voltage 18 24 26 V" or markdown "| 18 | 24 | 26 | V |"
TABLE_LIMITS_RE = re.compile(
    r"(?:(?P<name>[A-Za-z][A-Za-z /-]{2,48}):?[ \t]+)?"
    r"(?P<nums>(?:\$?-?\d[\d,]*(?:\.\d+)?[ \t]+){2,6})"
    r"(?P<unit>V|A|W|kW|MHz|GHz|USD|ms|mm|kg|C|°C|%)"
    r"(?=\s|$|[.,;:])"
)
SKIP_PARAM_NAMES = {"page", "figure", "table", "chapter", "section", "firmware"}


def extract_parameters(text: str, page: int, section_id: str | None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    flat = re.sub(r"[|]+", " ", text or "")

    def add(name: str, raw: str, unit: str) -> None:
        name = name.strip()
        if name.lower() in SKIP_PARAM_NAMES or name.lower().startswith("do not"):
            return
        unit = (unit or "").strip()
        raw_full = (raw + (f" {unit}" if unit else "")).strip()
        if name.startswith("metric_") and any(f["raw_string_value"] == raw_full for f in found):
            return
        key = f"{raw_full}|{name.lower()}"
        if key in seen:
            return
        seen.add(key)
        numeric = float(raw.replace("$", "").replace(",", ""))
        found.append(
            {
                "parameter_name": name[:80],
                "numeric_value": numeric,
                "raw_string_value": raw_full,
                "unit": "USD" if raw.startswith("$") else unit,
                "data_type": "currency" if raw.startswith("$") else "float",
                "provenance_page": page,
                "section_id": section_id,
            }
        )

    for m in NUMBER_RE.finditer(flat):
        add(m.group("name"), m.group("raw"), (m.group("unit") or "").strip())
    for m in BARE_METRIC_RE.finditer(flat):
        add(f"metric_{m.group('unit')}", m.group("raw"), m.group("unit"))
    for m in TABLE_LIMITS_RE.finditer(flat):
        name = (m.group("name") or f"metric_{m.group('unit')}").strip()
        unit = m.group("unit")
        for raw in m.group("nums").split():
            add(name, raw, unit)
    return found


def classify_document_intent(text: str, profiles: list[PageProfile], title: str = "") -> str:
    sample = ((title or "") + "\n" + text[:4000]).lower()
    table_pages = sum(1 for p in profiles if p.table_count > 0)
    academic_hits = sum(k in sample for k in ("abstract", "references", "et al", "doi", "related work", "technical report", "neurips"))
    financial_hits = sum(k in sample for k in ("revenue", "ebitda", "fiscal", "balance sheet", "units sold"))
    technical_hits = sum(k in sample for k in ("operating voltage", "datasheet", "pinout", "tolerance", "electrical ratings"))
    textbook_hits = sum(k in sample for k in ("chapter", "textbook", "exercise", "learning objectives"))
    if financial_hits >= 2 or ("$" in sample and financial_hits >= 1):
        return "financial"
    if technical_hits >= 2 or (technical_hits >= 1 and "datasheet" in (title or "").lower()):
        return "technical"
    if academic_hits >= 1 and academic_hits >= technical_hits:
        return "academic"
    if textbook_hits:
        return "textbook"
    if table_pages >= max(1, len(profiles) // 3) and financial_hits:
        return "financial"
    return "academic"
