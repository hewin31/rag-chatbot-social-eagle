import click
from src.embeddings.embeddings import generate_embeddings_for_document, query_document_index

@click.group()
def cli():
    pass

@cli.command()
@click.argument("document_id")
@click.option("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model spec. Use 'openai:MODEL' for OpenAI.")
@click.option("--threshold", default=0.85, help="Confidence threshold (0-1) for chunks to embed.")
@click.option("--batch", default=64, help="Batch size for embedding generation.")
def embed(document_id, model, threshold, batch):
    """Generate and persist embeddings for a document's trusted chunks."""
    res = generate_embeddings_for_document(document_id, model_name=model, confidence_threshold=threshold, batch_size=batch)
    click.echo(res)

@cli.command()
@click.argument("document_id")
@click.argument("query")
@click.option("--model", default="sentence-transformers/all-MiniLM-L6-v2")
@click.option("--topk", default=5)
def query(document_id, query, model, topk):
    res = query_document_index(document_id, query, top_k=topk, model_name=model)
    click.echo(res)

if __name__ == "__main__":
    cli()
