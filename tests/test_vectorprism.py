from app.vectorprism import CHANNELS, channel_texts, fuse, weights_for


def test_six_channels_always_emitted():
    texts = channel_texts(
        title="Nexus-24",
        intent="technical",
        section_path=["Electrical Ratings"],
        section_title="Electrical Ratings",
        body="Maximum Operating Voltage: 24 V",
        retrieval_text="Document: Nexus-24\nSection: Electrical Ratings\n\nMaximum Operating Voltage: 24 V",
        entities=["Nexus"],
        captions=["Table 2. Operating limits."],
        page_start=1,
    )
    assert set(texts) == set(CHANNELS)
    assert "24 V" in texts["numeric"]
    assert "Electrical Ratings" in texts["structural"]
    assert "Nexus" in texts["entity"]
    assert "Table 2" in texts["caption"]


def test_intent_moves_mass_to_numeric():
    param = weights_for("parameter")
    academic = weights_for("academic")
    assert abs(sum(param.values()) - 1.0) < 1e-9
    assert param["numeric"] > academic["numeric"]
    assert academic["semantic"] > param["semantic"]


def test_fuse_records_channels():
    hits = {
        "semantic": [{"chunk_id": "a", "text": "hello"}],
        "numeric": [{"chunk_id": "a", "text": "hello"}, {"chunk_id": "b", "text": "24 V"}],
    }
    for ch in CHANNELS:
        hits.setdefault(ch, [])
    merged, meta = fuse(hits, [], "parameter", k=4)
    assert meta["name"] == "VectorPrism"
    assert meta["channels"] == list(CHANNELS)
    assert merged[0]["chunk_id"] == "a"
    assert "numeric" in merged[0]["channels"]
