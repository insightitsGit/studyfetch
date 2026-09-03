from app.benchmark import generate_probes, heading_integrity, score_retrieval


def _hit(filename, text, **extra):
    row = {
        "filename": filename,
        "document_id": extra.pop("document_id", filename),
        "retrieval_text": text,
        "page_start": 1,
    }
    row.update(extra)
    return row


def test_generate_probes_follow_the_library_not_a_seed_script():
    docs = [
        {"id": "doc_a", "filename": "widget_manual.pdf", "title": "Widget-9 Manual", "page_count": 1},
        {"id": "doc_b", "filename": "garden_notes.pdf", "title": "Garden Notes", "page_count": 1},
    ]
    pages = [
        {
            "document_id": "doc_a",
            "page_number": 1,
            "text_preview": (
                "2 Electrical Ratings\n"
                "Maximum Operating Voltage: 48 V\n"
                "Peak Power: 12 W\n"
                "The widget firmware uses a Kalman filter on the encoder ticks."
            ),
            "label": "table_heavy",
        },
        {
            "document_id": "doc_b",
            "page_number": 1,
            "text_preview": (
                "Chapter 4 The fox and the baobabs\n"
                "The fox asked to be tamed. Baobabs threaten the planet if left to sprout."
            ),
            "label": "digital_text",
        },
    ]
    probes = generate_probes(docs, pages, [])
    kinds = {p["kind"] for p in probes}
    queries = " ".join(p["query"] for p in probes).lower()
    assert probes
    assert "parameter" in kinds
    assert any(p.get("document_id") == "doc_a" and "48 V" in (p.get("must_contain") or []) for p in probes)
    assert any("fox" in (p.get("must_contain") or [""])[0].lower() or "baobab" in p["query"].lower() for p in probes)
    assert "prince" not in queries
    assert "nexus24" not in queries
    assert not any(p["id"].startswith("q_voltage") for p in probes)

    extra_doc = {"id": "doc_c", "filename": "orbit.pdf", "title": "Orbit Primer", "page_count": 1}
    extra_page = {
        "document_id": "doc_c",
        "page_number": 1,
        "text_preview": "Apoapsis is the farthest point. Apoapsis heating peaks at reentry.",
        "label": "digital_text",
    }
    more = generate_probes(docs + [extra_doc], pages + [extra_page], [])
    assert any("apoapsis" in p["query"].lower() or "Apoapsis" in (p.get("must_contain") or []) for p in more)


def test_cross_document_probe_requires_shared_owners():
    docs = [
        {"id": "a", "filename": "paper.pdf", "title": "Attention Routing", "page_count": 1},
        {"id": "b", "filename": "sheet.pdf", "title": "Controller Sheet", "page_count": 1},
    ]
    pages = [
        {"document_id": "a", "page_number": 1, "text_preview": "Encoder attention routing for study systems.", "label": "digital_text"},
        {"document_id": "b", "page_number": 1, "text_preview": "Firmware attention routing on the controller.", "label": "digital_text"},
    ]
    probes = generate_probes(docs, pages, [])
    cross = [p for p in probes if p.get("kind") == "cross_document"]
    assert cross
    assert set(cross[0]["require_document_ids"]) == {"a", "b"}


def test_hit_at_1_uses_retrieval_text_not_body():
    gold = {
        "must_contain": ["attention"],
        "prefer_contain": ["encoder"],
        "cross_doc": True,
        "require_document_ids": ["doc_paper", "doc_sheet"],
        "avoid_document_ids": ["doc_novel"],
    }
    payload = {
        "pipeline_id": "prism",
        "vectorprism": {
            "channels": [
                "semantic",
                "structural",
                "title",
                "entity",
                "numeric",
                "caption",
            ],
            "weights": {"numeric": 0.1, "semantic": 0.4},
        },
        "hits": [
            _hit(
                "attention_routing_seed.pdf",
                "Document: Attention Routing\n\nEncoder and decoder stacks.",
                document_id="doc_paper",
                channels=["semantic"],
            ),
            _hit(
                "nexus24_datasheet_seed.pdf",
                "Document: Nexus-24\n\nFirmware notes. Encoder attention in the industrial controller.",
                document_id="doc_sheet",
                channels=["chorusgraph"],
                graph_edge="overlaps_with",
            ),
        ],
    }
    quality = score_retrieval(gold, payload)
    names = {c["name"]: c["pass"] for c in quality["checks"]}
    assert names["hit_at_1"] is True
    assert names["vectorprism_6ch"] is True
    assert names["chorusgraph_related"] is True
    assert names["cross_document"] is True
    assert quality["score"] > 70


def test_distractor_first_is_not_a_perfect_score():
    gold = {
        "must_contain": ["48 V"],
        "prefer_contain": ["Maximum Operating Voltage"],
        "document_id": "doc_manual",
        "avoid_document_ids": ["doc_novel"],
    }
    good = _hit("widget_manual.pdf", "Maximum Operating Voltage: 48 V", document_id="doc_manual")
    bad = _hit("garden_notes.pdf", "The fox asked to be tamed.", document_id="doc_novel")
    perfect = score_retrieval(gold, {"pipeline_id": "baseline", "hits": [good, good]})
    distracted = score_retrieval(gold, {"pipeline_id": "baseline", "hits": [bad, good]})
    assert perfect["score"] > distracted["score"]
    assert distracted["mrr"] < perfect["mrr"]
    assert distracted["ndcg"] < perfect["ndcg"]
    assert perfect["score"] < 100


def test_heading_integrity_does_not_need_a_named_pdf():
    weak = heading_integrity([{"title": "Untitled", "level": 1, "page_start": 1}], 4)
    strong = heading_integrity(
        [
            {"title": "2 Electrical Ratings", "level": 1, "page_start": 1},
            {"title": "2.1 Limits", "level": 2, "page_start": 1},
            {"title": "3 Safety", "level": 1, "page_start": 2},
        ],
        2,
    )
    assert strong > weak
    assert strong > 0.7
