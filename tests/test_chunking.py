"""Light tests for chunking module."""


def test_chunking_imports():
    """Ensure chunking module imports correctly."""
    from src.ingest.chunking import adaptive_chunk, chunk_text_semantic, chunk_table, estimate_tokens
    assert adaptive_chunk is not None
    assert chunk_text_semantic is not None
    assert chunk_table is not None
    assert estimate_tokens is not None


def test_estimate_tokens():
    """Test token estimation."""
    from src.ingest.chunking import estimate_tokens
    text = "hello world this is a test"
    tokens = estimate_tokens(text)
    assert tokens == 6
