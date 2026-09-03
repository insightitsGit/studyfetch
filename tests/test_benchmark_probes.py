from app.benchmark import generate_probes, running_headers


def test_empty_library_has_no_probes():
    assert generate_probes([], [], []) == []


def test_running_headers_are_detected_from_repeat_lines():
    pages = [
        {"text_preview": "Acme Manual\nBody one\nPage 1"},
        {"text_preview": "Acme Manual\nBody two\nPage 2"},
        {"text_preview": "Acme Manual\nBody three\nPage 3"},
    ]
    headers = running_headers(pages)
    assert "acme manual" in headers
