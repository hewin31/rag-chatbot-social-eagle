def test_imports():
    import src.embeddings.embeddings as emb
    assert hasattr(emb, 'generate_embeddings_for_document')


def test_cli_import():
    import src.embeddings.cli as cli
    assert hasattr(cli, 'cli')
