from app.extract.pdf_common import TextBlock, attach_captions, extract_images, open_pdf
from app.seed import write_field_note_pdf


def test_field_note_embeds_a_detectable_raster_figure(tmp_path):
    path = tmp_path / "nexus24_attention_field_note_seed.pdf"
    write_field_note_pdf(path)
    doc = open_pdf(str(path))
    try:
        images = doc[0].get_images(full=True)
        assets = extract_images(doc, "doc_field")
    finally:
        doc.close()
    assert images, "PyMuPDF must see an embedded raster (vector drawings are invisible here)"
    figures = [a for a in assets if a.get("asset_type") == "figure" and a.get("blob_uri")]
    assert figures, "Outline reads figure assets with a blob_uri"
    assert all(len((a.get("blob_uri") or "")) > 0 for a in figures)


def test_attach_captions_binds_figure_line_not_the_following_table():
    assets = [
        {"asset_type": "figure", "page_number": 1, "caption": ""},
        {"asset_type": "table", "page_number": 1, "caption": ""},
    ]
    blocks = [
        TextBlock(1, "Figure 2. Panel photo.\nTable 3. Weights.", 0, 0, 10, 10, 9, "Times", False),
        TextBlock(1, "Table 3. Weights.", 0, 20, 10, 30, 9, "Times", False),
    ]
    attach_captions(assets, blocks)
    assert assets[0]["caption"] == "Figure 2. Panel photo."
    assert assets[1]["caption"] == "Table 3. Weights."
