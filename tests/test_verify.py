"""Light tests for verification module."""


def test_verify_imports():
    """Ensure verification modules import correctly."""
    from src.verify.sql_audit import audit_document, list_documents_status
    from src.verify.cross_check import cross_check_document
    assert audit_document is not None
    assert list_documents_status is not None
    assert cross_check_document is not None
