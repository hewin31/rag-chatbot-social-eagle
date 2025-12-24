"""CLI for Phase 4 verification and audit."""

import click
import json
from .sql_audit import audit_document, list_documents_status
from .cross_check import cross_check_document


@click.group()
def cli():
    pass


@cli.command()
@click.argument('document_id')
def audit(document_id):
    """Run SQL audit on a document.

    Shows block counts, confidence stats, extraction methods, and any issues.
    """
    report = audit_document(document_id)
    click.echo(json.dumps(report, indent=2, default=str))


@cli.command()
@click.argument('document_id')
@click.option('--samples', default=3, help='Number of blocks to sample for validation')
def validate(document_id, samples):
    """Cross-check extracted blocks against source PDF.

    Samples blocks from DB and re-extracts to compare (allows ±10% variance).
    """
    report = cross_check_document(document_id, sample_size=samples)
    click.echo(json.dumps(report, indent=2, default=str))


@cli.command()
def status():
    """List all documents and their ingestion status."""
    docs = list_documents_status()
    click.echo(json.dumps(docs, indent=2, default=str))


if __name__ == '__main__':
    cli()
