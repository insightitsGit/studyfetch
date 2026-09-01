from app.benchmark import score_retrieval


def _hit(filename, text, **extra):
    row = {
        "filename": filename,
        "document_id": filename,
        "retrieval_text": text,
        "page_start": 1,
    }
    row.update(extra)
    return row


def test_hit_at_1_uses_retrieval_text_not_body():
    gold = {
        "must_contain": ["attention"],
        "prefer_contain": ["encoder"],
        "cross_doc": True,
        "require_files": ["attention", "nexus24"],
        "avoid_filename": "prince",
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
                channels=["semantic"],
            ),
            _hit(
                "nexus24_datasheet_seed.pdf",
                "Document: Nexus-24\n\nFirmware notes. Encoder attention in the industrial controller.",
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
        "must_contain": ["24 V"],
        "prefer_contain": ["Maximum Operating Voltage"],
        "filename_hint": "nexus24",
        "avoid_filename": "prince",
    }
    good = _hit("nexus24_datasheet_seed.pdf", "Maximum Operating Voltage: 24 V")
    bad = _hit("the_little_prince.pdf", "The little prince watched the sunset.")
    perfect = score_retrieval(gold, {"pipeline_id": "baseline", "hits": [good, good]})
    distracted = score_retrieval(gold, {"pipeline_id": "baseline", "hits": [bad, good]})
    assert perfect["score"] > distracted["score"]
    assert distracted["mrr"] < perfect["mrr"]
    assert distracted["ndcg"] < perfect["ndcg"]
    assert perfect["score"] < 100
