from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line

from app.config import settings


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="DocTitle", parent=ss["Title"], fontSize=20, spaceAfter=12))
    ss.add(ParagraphStyle(name="H1", parent=ss["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=8))
    ss.add(ParagraphStyle(name="H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6))
    ss.add(ParagraphStyle(name="Body", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=8))
    ss.add(ParagraphStyle(name="Caption", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#334155")))
    ss.add(ParagraphStyle(name="Footer", parent=ss["Normal"], fontSize=8, textColor=colors.grey))
    return ss


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Italic", 8)
    canvas.drawString(inch, letter[1] - 0.45 * inch, doc.title)
    canvas.drawRightString(letter[0] - inch, letter[1] - 0.45 * inch, "Studyfetch seed corpus")
    canvas.drawString(inch, 0.45 * inch, "Confidential draft — headers repeat on every page")
    canvas.drawRightString(letter[0] - inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _attention_figure() -> Drawing:
    d = Drawing(400, 140)
    d.add(Rect(10, 20, 90, 90, fillColor=colors.HexColor("#dbeafe"), strokeColor=colors.HexColor("#1d4ed8")))
    d.add(String(30, 60, "Encoder"))
    d.add(Rect(160, 20, 90, 90, fillColor=colors.HexColor("#dcfce7"), strokeColor=colors.HexColor("#15803d")))
    d.add(String(180, 60, "Decoder"))
    d.add(Line(100, 65, 160, 65, strokeColor=colors.HexColor("#0f172a"), strokeWidth=2))
    d.add(String(110, 80, "Attention"))
    d.add(Rect(300, 40, 80, 50, fillColor=colors.HexColor("#fef3c7"), strokeColor=colors.HexColor("#b45309")))
    d.add(String(312, 60, "Output"))
    d.add(Line(250, 65, 300, 65, strokeColor=colors.HexColor("#0f172a"), strokeWidth=2))
    return d


def write_academic_pdf(path: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        title="Attention Routing in Multi-Column Study Systems",
        author="Studyfetch Seed Lab",
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
    )
    story = [
        Paragraph("Attention Routing in Multi-Column Study Systems", styles["DocTitle"]),
        Paragraph(
            "Jane Ortiz, Priya Nair, and Samuel Chen. Seed University Technical Report 2024.",
            styles["Body"],
        ),
        Paragraph("1 Introduction", styles["H1"]),
        Paragraph(
            "Large collections of educational PDFs mix digitally generated textbooks, research papers, "
            "and scanned worksheets. A useful retrieval system must reconstruct section hierarchy, keep "
            "page-level provenance, and avoid treating running headers as content. This paper describes "
            "an encoder-decoder attention stack used as a teaching example for document intelligence.",
            styles["Body"],
        ),
        Paragraph("2 Related Work", styles["H1"]),
        Paragraph(
            "Prior pipelines dump raw PDF text into sliding windows. That approach collapses multi-column "
            "layouts and loses the difference between a figure caption and the surrounding paragraph. "
            "Layout-aware parsers that cluster x-coordinates restore reading order before chunking.",
            styles["Body"],
        ),
        Paragraph("3 Model Architecture", styles["H1"]),
        Paragraph("3.1 Encoder and Decoder Stacks", styles["H2"]),
        Paragraph(
            "The encoder maps each section into a contextual representation. The decoder attends over "
            "those representations when answering a study question. Residual connections stabilize training. "
            "We write the core update as Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, which must be "
            "preserved as notation rather than rewritten into prose.",
            styles["Body"],
        ),
        Paragraph("3.2 Multi-Head Routing", styles["H2"]),
        Paragraph(
            "Different heads specialize: one tracks headings, one tracks tables, one tracks figure captions. "
            "This is the pedagogical analogue of page-type routing in a document pipeline. Pages with usable "
            "digital text should not pay for a multimodal model.",
            styles["Body"],
        ),
        KeepTogether(
            [
                _attention_figure(),
                Paragraph("Figure 1. Encoder-decoder attention used as a teaching diagram.", styles["Caption"]),
            ]
        ),
        Paragraph("4 Experiments", styles["H1"]),
        Paragraph(
            "We evaluate heading recovery, table isolation, and retrieval of the exact phrase "
            "\"page-level provenance\". Sliding-window baselines retrieve neighboring boilerplate. "
            "Section-aware chunks retrieve the methods subsection.",
            styles["Body"],
        ),
        Paragraph("Table 1. Toy retrieval accuracy on the seed corpus.", styles["Caption"]),
    ]
    data = [
        ["System", "Heading F1", "Table exact", "Latency"],
        ["Sliding window", "0.61", "0.40", "40 ms"],
        ["Section chunks", "0.84", "0.73", "48 ms"],
        ["Graph + parameters", "0.81", "0.88", "62 ms"],
    ]
    table = Table(data, colWidths=[2.1 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [
        table,
        Spacer(1, 12),
        Paragraph("5 Conclusion", styles["H1"]),
        Paragraph(
            "Document intelligence is a routing problem. Reconstruct structure first, embed second, and "
            "keep every claim traceable to a page. The accompanying datasheet shows how the same attention "
            "stack is referenced in an industrial controller manual — a cross-document overlap.",
            styles["Body"],
        ),
        Paragraph("References", styles["H1"]),
        Paragraph("[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.", styles["Body"]),
        Paragraph("[2] Studyfetch. Document Intelligence Assignment Brief. 2026.", styles["Body"]),
    ]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def write_datasheet_pdf(path: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        title="Nexus-24 Industrial Controller Datasheet",
        author="Nexus Instruments",
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
    )
    story = [
        Paragraph("Nexus-24 Industrial Controller Datasheet", styles["DocTitle"]),
        Paragraph("Nexus Instruments — Fiscal Q4 technical bulletin.", styles["Body"]),
        Paragraph("1 Overview", styles["H1"]),
        Paragraph(
            "The Nexus-24 is a panel-mount controller for lab and factory lines. Firmware 3.2 adds an "
            "optional transformer attention module so operators can query runbooks stored as PDFs. "
            "This datasheet is deliberately numeric: downstream agents must not invent voltages.",
            styles["Body"],
        ),
        Paragraph("2 Electrical Ratings", styles["H1"]),
        Paragraph("Maximum Operating Voltage: 24 V", styles["Body"]),
        Paragraph("Nominal Input Current: 1.5 A", styles["Body"]),
        Paragraph("Peak Power: 36 W", styles["Body"]),
        Paragraph("Isolation Tolerance: 1500 V", styles["Body"]),
        Paragraph("Watchdog Timeout: 250 ms", styles["Body"]),
        Paragraph("3 Commercial Metrics", styles["H1"]),
        Paragraph("Q4 Revenue: $1,042,500", styles["Body"]),
        Paragraph("Units Sold: 875", styles["Body"]),
        Paragraph("Average Selling Price: $1192", styles["Body"]),
        Paragraph("Table 2. Operating limits.", styles["Caption"]),
    ]
    data = [
        ["Parameter", "Min", "Typ", "Max", "Unit"],
        ["Operating Voltage", "18", "24", "26", "V"],
        ["Input Current", "0.8", "1.5", "2.0", "A"],
        ["Ambient Temperature", "0", "25", "50", "C"],
        ["Watchdog Timeout", "100", "250", "400", "ms"],
    ]
    table = Table(data, colWidths=[1.8 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#6b7280")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f4f6")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story += [
        table,
        Paragraph("4 Firmware Notes", styles["H1"]),
        Paragraph(
            "Runbook retrieval uses the same encoder-decoder attention stack described in the academic "
            "companion paper. Queries about industrial controllers should surface both this datasheet "
            "and the section on multi-head routing. Do not substitute 24 V with a neighboring 26 V "
            "from a vector neighbor.",
            styles["Body"],
        ),
        Paragraph("5 Safety", styles["H1"]),
        Paragraph(
            "Do not exceed Maximum Operating Voltage: 24 V. Isolation Tolerance: 1500 V is a type-test "
            "value, not a continuous rating. Q4 Revenue: $1,042,500 is the audited figure for this bulletin.",
            styles["Body"],
        ),
    ]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def _nexus_panel_png() -> BytesIO:
    """Raster figure so Outline can show a detected image (vector drawings do not extract)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    img = PILImage.new("RGB", (860, 400), "#0b1220")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        body_font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()
    draw.rounded_rectangle((24, 24, 836, 376), radius=22, fill="#1e293b", outline="#38bdf8", width=4)
    draw.rounded_rectangle((56, 56, 400, 250), radius=10, fill="#020617", outline="#22c55e", width=3)
    draw.text((76, 76), "NEXUS-24", fill="#e2e8f0", font=title_font)
    draw.text((76, 124), "HMI  ·  Firmware 3.2", fill="#93c5fd", font=body_font)
    draw.text((76, 176), "FIELD DERATE   22 V", fill="#86efac", font=title_font)
    draw.text((76, 220), "ISO CONTINUOUS   600 V", fill="#fcd34d", font=body_font)
    draw.rounded_rectangle((440, 56, 800, 250), radius=10, fill="#312e81", outline="#c4b5fd", width=3)
    draw.text((460, 76), "ENCODER", fill="#f5f3ff", font=title_font)
    draw.text((460, 124), "attention module", fill="#ddd6fe", font=body_font)
    draw.text((460, 168), "Q  →  K  →  V", fill="#fde68a", font=body_font)
    draw.text((460, 208), "behind the panel HMI", fill="#c4b5fd", font=small_font)
    draw.rectangle((56, 280, 800, 344), fill="#0f172a", outline="#64748b", width=2)
    draw.text((76, 300), "Photo plate  ·  field commissioning  ·  AN-24-07", fill="#cbd5e1", font=body_font)
    buf = BytesIO()
    buf.name = "nexus_panel.png"
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def write_field_note_pdf(path: Path) -> None:
    """Third seed: same world as the paper + datasheet, built to show VectorPrism 6ch."""
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        title="Nexus-24 Application Note AN-24-07 — Attention Module Field Commissioning",
        author="Nexus Instruments Field Engineering",
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
    )
    mix = [
        ["Channel", "Role on the panel", "Parameter-intent weight"],
        ["semantic", "Runbook prose", "0.16"],
        ["structural", "Heading path 3.1 Field Encoder", "0.18"],
        ["title", "AN-24-07 / Field Derate", "0.10"],
        ["entity", "Nexus-24, Ortiz, Firmware 3.2", "0.08"],
        ["numeric", "22 V derate, not 24 V typ", "0.38"],
        ["caption", "Figure 2 six-channel mix", "0.10"],
    ]
    mix_table = Table(mix, colWidths=[1.3 * inch, 3.1 * inch, 1.8 * inch])
    mix_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c1d95")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a78bfa")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f3ff")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    derate = [
        ["Parameter", "Min", "Typ", "Max", "Unit"],
        ["Field Derate Voltage", "20", "22", "24", "V"],
        ["Continuous Isolation", "400", "600", "800", "V"],
        ["Attention Inference Budget", "8", "12", "16", "ms"],
    ]
    derate_table = Table(derate, colWidths=[2.2 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch])
    derate_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14532d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#86efac")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0fdf4")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story = [
        Paragraph(
            "Nexus-24 Application Note AN-24-07 — Attention Module Field Commissioning",
            styles["DocTitle"],
        ),
        Paragraph(
            "Nexus Instruments Field Engineering. Companion to the Ortiz / Nair / Chen attention-routing "
            "technical report and the Nexus-24 Q4 datasheet.",
            styles["Body"],
        ),
        Paragraph("1 Purpose", styles["H1"]),
        Paragraph(
            "This note commissions the optional transformer attention module on a Nexus-24 industrial "
            "controller running Firmware 3.2. It is the field document. The datasheet lists factory "
            "ratings. The academic paper describes the encoder-decoder attention stack. Operators who "
            "ask one index a generic “voltage” question will mix these three files. VectorPrism keeps "
            "the subspaces apart.",
            styles["Body"],
        ),
        Paragraph("2 Why a six-channel mix", styles["H1"]),
        Paragraph(
            "A single semantic channel treats Field Derate Voltage: 22 V as a neighbor of Maximum "
            "Operating Voltage: 24 V and of the table-max 26 V. The numeric channel must win parameter "
            "questions. The caption channel must win “show me the six-channel figure.” The entity "
            "channel must keep Nexus-24, Ortiz, and Firmware 3.2 on this note or the paper — not invent "
            "a third controller.",
            styles["Body"],
        ),
        KeepTogether(
            [
                Image(_nexus_panel_png(), width=5.8 * inch, height=2.7 * inch),
                Paragraph(
                    "Figure 2. Nexus-24 attention module behind the HMI (field derate 22 V).",
                    styles["Caption"],
                ),
            ]
        ),
        Paragraph("Table 3. Parameter-intent channel weights used on the panel retriever.", styles["Caption"]),
        mix_table,
        Spacer(1, 10),
        Paragraph("3 Field Encoder Path", styles["H1"]),
        Paragraph("3.1 Field Encoder Path", styles["H2"]),
        Paragraph(
            "The encoder maps the commissioned runbook the same way section 3.1 of the academic paper "
            "maps a study PDF: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. Head Count: 6. "
            "Embedding Dimension: 384. Multi-head routing on the panel is the industrial version of "
            "the paper's heading / table / caption heads. Do not retrieve the paper's toy latency "
            "row (40 ms / 48 ms / 62 ms) as the field budget.",
            styles["Body"],
        ),
        Paragraph("3.2 Commissioned electrical setpoints", styles["H2"]),
        Paragraph("Field Derate Voltage: 22 V", styles["Body"]),
        Paragraph("Continuous Isolation: 600 V", styles["Body"]),
        Paragraph("Attention Inference Budget: 12 ms", styles["Body"]),
        Paragraph("Head Count: 6", styles["Body"]),
        Paragraph("Embedding Dimension: 384", styles["Body"]),
        Paragraph("Table 4. Field derate limits (not the datasheet factory table).", styles["Caption"]),
        derate_table,
        Spacer(1, 10),
        Paragraph("4 Cross-document map", styles["H1"]),
        Paragraph(
            "Queries about the encoder-decoder attention stack used in industrial controllers should "
            "surface this note, the paper's 3.1 Encoder and Decoder Stacks, and the datasheet Firmware "
            "Notes. That is ChorusGraph, not a bigger chunk. Q4 Revenue: $1,042,500 stays on the "
            "datasheet — this bulletin has no commercial metrics.",
            styles["Body"],
        ),
        Paragraph("5 Safety", styles["H1"]),
        Paragraph(
            "Do not set the panel to Maximum Operating Voltage: 24 V. That is the datasheet typical. "
            "The commissioned setpoint is Field Derate Voltage: 22 V. Isolation Tolerance: 1500 V is "
            "the factory type-test. Continuous Isolation: 600 V is what the field encoder path is "
            "rated for after derate. Neighbor digits in Table 4 (20 V / 24 V) are not the typ.",
            styles["Body"],
        ),
    ]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


SEED_REVISION = "raster-figure-2"
FIELD_NOTE_NAME = "nexus24_attention_field_note_seed.pdf"


def ensure_seed_pdfs() -> list[Path]:
    settings.ensure_dirs()
    marker = settings.seed_dir / ".revision"
    stale = (not marker.exists()) or marker.read_text(encoding="utf-8").strip() != SEED_REVISION
    specs = [
        (settings.seed_dir / "attention_routing_seed.pdf", write_academic_pdf),
        (settings.seed_dir / "nexus24_datasheet_seed.pdf", write_datasheet_pdf),
        (settings.seed_dir / FIELD_NOTE_NAME, write_field_note_pdf),
    ]
    paths = []
    for path, writer in specs:
        if not path.exists() or (stale and path.name == FIELD_NOTE_NAME):
            writer(path)
        paths.append(path)
    if stale:
        marker.write_text(SEED_REVISION, encoding="utf-8")
    return paths


def seed_corpus(store) -> list[dict]:
    from app.pipelines.base import ingest_pdf
    from app.storage.blobs import sha256_bytes

    paths = ensure_seed_pdfs()
    existing = {d["filename"]: d for d in store.fetchall("SELECT * FROM documents")}
    docs = []
    for path in paths:
        data = path.read_bytes()
        digest = sha256_bytes(data)
        old = existing.get(path.name)
        if old and old.get("sha256") == digest:
            docs.append(old)
            continue
        if old:
            store.delete_document(old["id"])
        docs.append(ingest_pdf(store, path.name, data))
    known = {d["filename"] for d in docs}
    for doc in existing.values():
        if doc["filename"] not in known:
            docs.append(doc)
    return docs
