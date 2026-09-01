from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line

from app.config import settings
from app.db.store import Store
from app.pipelines.base import ingest_pdf


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


def ensure_seed_pdfs() -> list[Path]:
    settings.ensure_dirs()
    academic = settings.seed_dir / "attention_routing_seed.pdf"
    datasheet = settings.seed_dir / "nexus24_datasheet_seed.pdf"
    if not academic.exists():
        write_academic_pdf(academic)
    if not datasheet.exists():
        write_datasheet_pdf(datasheet)
    return [academic, datasheet]


def seed_corpus(store: Store) -> list[dict]:
    ensure_seed_pdfs()
    existing = store.fetchall("SELECT * FROM documents")
    if existing:
        return existing
    docs = []
    for path in ensure_seed_pdfs():
        data = path.read_bytes()
        doc = ingest_pdf(store, path.name, data)
        docs.append(doc)
    return docs
