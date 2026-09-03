from app.retrieve import boost_hits_with_signed_params, param_name_overlap


def test_commission_question_matches_field_name_not_factory_name():
    q = "What voltage should I commission on the Nexus-24?"
    assert param_name_overlap(q, "Field Commissioning Voltage") >= 2
    assert param_name_overlap(q, "Maximum Operating Voltage") < 2


def test_factory_question_matches_operating_name():
    q = "What is the Maximum Operating Voltage?"
    assert param_name_overlap(q, "Maximum Operating Voltage") >= 2
    assert param_name_overlap(q, "Field Commissioning Voltage") < 2


def test_boost_reranks_the_signed_match_first():
    class FakeStore:
        def fetchall(self, sql, params=()):
            return [
                {
                    "document_id": "factory",
                    "parameter_name": "Maximum Operating Voltage",
                    "raw_string_value": "24 V",
                    "provenance_page": 1,
                },
                {
                    "document_id": "field",
                    "parameter_name": "Field Commissioning Voltage",
                    "raw_string_value": "22 V",
                    "provenance_page": 2,
                },
            ]

    hits = [
        {"chunk_id": "a", "document_id": "factory", "score": 0.9, "page_start": 1, "channels": []},
        {"chunk_id": "b", "document_id": "field", "score": 0.4, "page_start": 2, "channels": []},
    ]
    out = boost_hits_with_signed_params(
        FakeStore(), hits, "What voltage should I commission on the Nexus-24?"
    )
    assert out[0]["document_id"] == "field"
    assert out[0]["param_boost"]["value"] == "22 V"
    assert "signed_param" in out[0]["channels"]
