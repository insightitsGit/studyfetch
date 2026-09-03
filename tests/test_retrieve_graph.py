from app.retrieve import merge_graph_hits


def test_duplicate_graph_neighbors_do_not_keyerror():
    merged = [
        {
            "chunk_id": "chk_a",
            "section_id": "sec_1",
            "document_id": "d1",
            "channels": ["semantic"],
            "retrieval_text": "encoder attention",
        },
        {
            "chunk_id": "chk_b",
            "section_id": "sec_2",
            "document_id": "d1",
            "channels": ["title"],
            "retrieval_text": "firmware notes",
        },
    ]
    extra = [
        {
            "id": "chk_c",
            "chunk_id": "chk_c",
            "document_id": "d2",
            "graph_edge": "overlaps_with",
            "graph_weight": 0.71,
            "retrieval_text": "industrial controller",
        },
        {
            "id": "chk_c",
            "chunk_id": "chk_c",
            "document_id": "d2",
            "graph_edge": "overlaps_with",
            "graph_weight": 0.77,
            "retrieval_text": "industrial controller",
        },
        {
            "chunk_id": "chk_a",
            "document_id": "d1",
            "graph_edge": "same_entity",
            "graph_weight": 0.9,
        },
    ]
    out = merge_graph_hits(merged, extra)
    ids = [h["chunk_id"] for h in out]
    assert ids.count("chk_c") == 1
    related = next(h for h in out if h["chunk_id"] == "chk_c")
    assert "chorusgraph" in related["channels"]
    assert related["graph_weight"] == 0.77
    source = next(h for h in out if h["chunk_id"] == "chk_a")
    assert "chorusgraph" in source["channels"]
    assert source["graph_edge"] == "same_entity"
