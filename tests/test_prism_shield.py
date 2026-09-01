"""PrismShield classifies signed vs drifted metrics; it does not rewrite prose."""

from app.extract.pdf_common import extract_parameters
from app.pipelines.prism import PrismShield


class _Store:
    def __init__(self, params):
        self._params = params

    def fetchall(self, *_a, **_k):
        return self._params


def _params(*raws):
    rows = []
    for raw in raws:
        found = extract_parameters(raw if ":" in raw or " " in raw else f"Metric: {raw}", 1, None)
        if not found:
            found = extract_parameters(f"Value: {raw}", 1, None)
        for p in found:
            p["manifest_signature"] = "sig"
            rows.append(p)
    return rows


def test_signed_voltage_stays_in_text():
    store = _Store(_params("Maximum Operating Voltage: 24 V", "Operating Voltage 18 24 26 V"))
    text = "Maximum Operating Voltage: 24 V. Table max 26 V. Isolation 1500 V is signed separately."
    out = PrismShield(store).filter_chunks([{"document_id": "d", "text": text}])[0]
    assert "24 V" in out["retrieval_text"]
    assert "[UNVERIFIED]" not in out["retrieval_text"]
    assert "24 V" in out["shield"]["verified_parameters"]
    assert "26 V" in out["shield"]["verified_parameters"]


def test_invented_neighbor_is_drift_not_rewrite():
    store = _Store(_params("Maximum Operating Voltage: 24 V", "Operating Voltage 18 24 26 V"))
    text = "Use 23 V instead of the signed 24 V rating."
    out = PrismShield(store).filter_chunks([{"document_id": "d", "text": text}])[0]
    assert "23 V" in out["retrieval_text"]
    assert any(d["raw"] == "23 V" for d in out["shield"]["drifted"])
    assert "24 V" in out["shield"]["verified_parameters"]
    rewritten = PrismShield(store).filter_chunks(
        [{"document_id": "d", "text": text}], rewrite_drift=True
    )[0]
    assert "[DRIFT:23 V]" in rewritten["retrieval_text"]
    assert "24 V" in rewritten["retrieval_text"]


def test_academic_prose_is_unsigned():
    store = _Store(_params("Maximum Operating Voltage: 24 V"))
    text = "Section 3.1 reports Heading F1 0.61 in 2024 with 8 encoder heads."
    out = PrismShield(store).filter_chunks([{"document_id": "d", "text": text}])[0]
    assert "0.61" in out["retrieval_text"]
    assert "3.1" in out["retrieval_text"]
    assert "2024" in out["retrieval_text"]
    assert out["shield"]["drifted"] == []
    assert "0.61" in out["shield"]["unsigned"]
    assert "2024" in out["shield"]["unsigned"]


def test_query_payload_rolls_up_shield_summary():
    from app.retrieve import _shield_summary

    hits = [
        {
            "shield": {
                "verified_parameters": ["24 V"],
                "unsigned": ["2024"],
                "drifted": [{"raw": "23 V"}],
            }
        }
    ]
    summary = _shield_summary(hits, True)
    assert summary["applied"] is True
    assert "24 V" in summary["verified"]
    assert "2024" in summary["unsigned"]
    assert "23 V" in summary["drifted"]


def test_table_limits_are_extracted():
    found = extract_parameters("Operating Voltage 18 24 26 V\nWatchdog Timeout 100 250 400 ms", 1, None)
    volts = {round(p["numeric_value"], 3) for p in found if (p.get("unit") or "").upper() == "V"}
    ms = {round(p["numeric_value"], 3) for p in found if (p.get("unit") or "").lower() == "ms"}
    assert {18.0, 24.0, 26.0} <= volts
    assert {100.0, 250.0, 400.0} <= ms
