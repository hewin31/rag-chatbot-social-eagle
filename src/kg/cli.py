"""CLI for knowledge graph extraction and queries."""
import click
import json
from src.kg.extraction import extract_kg_for_document, sync_kg_to_neo4j
from src.kg.neo4j_driver import Neo4jDriver
from src.db import get_session
from src.db.models import Entity, Relationship
from src.config import logger


@click.group()
def kg_cli():
    """Knowledge Graph operations."""
    pass


@kg_cli.command()
@click.argument("document_id")
@click.option("--threshold", default=85, help="Confidence threshold for chunks (0-100)")
@click.option("--model", default="gpt-3.5-turbo", help="LLM model name")
def extract(document_id, threshold, model):
    """Extract entities and relationships from a document."""
    click.echo(f"Extracting KG for document {document_id}...")
    result = extract_kg_for_document(document_id, confidence_threshold=threshold, model=model)
    click.echo(json.dumps(result, indent=2))


@kg_cli.command()
@click.argument("document_id")
@click.option("--neo4j-uri", default=None, help="Neo4j URI (default: NEO4J_URI env var)")
def sync(document_id, neo4j_uri):
    """Sync SQL KG to Neo4j."""
    click.echo(f"Syncing KG to Neo4j for document {document_id}...")
    driver = Neo4jDriver(uri=neo4j_uri)
    if not driver.driver:
        click.echo("Failed to connect to Neo4j")
        return
    result = sync_kg_to_neo4j(document_id, driver)
    click.echo(json.dumps(result, indent=2, default=str))
    driver.close()


@kg_cli.command()
@click.argument("document_id")
def status(document_id):
    """Show KG statistics for a document."""
    with get_session() as session:
        entity_count = session.query(Entity).filter(Entity.document_id == document_id).count()
        rel_count = session.query(Relationship).filter(Relationship.document_id == document_id).count()
        
        click.echo(f"Document {document_id}:")
        click.echo(f"  Entities: {entity_count}")
        click.echo(f"  Relationships: {rel_count}")


if __name__ == "__main__":
    kg_cli()