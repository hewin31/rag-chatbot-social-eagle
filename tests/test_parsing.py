"""Light tests for parsing module."""


def test_parsing_imports():
    """Ensure parsing module imports correctly."""
    from src.ingest.parsing import extract_text_from_page, extract_tables_from_page, parse_document, classify_page_type
    assert extract_text_from_page is not None
    assert extract_tables_from_page is not None
    assert parse_document is not None
    assert classify_page_type is not None
