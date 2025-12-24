import click
from .probe import probe_document
from ..db import get_session
from ..db.models import Document, IngestionStatus
from pathlib import Path
import json


@click.group()
def cli():
    pass


@cli.command()
@click.argument('target')
@click.option('--samples', default=3, help='Number of pages to sample')
def probe(target, samples):
    """Run probe on a PDF file path or on a registered document_id.

    If `target` is a path to a PDF, the probe runs and prints JSON summary.
    If `target` matches a document_id in the DB, the probe runs and persists
    the probe_summary into `documents.probe_summary` and sets ingestion_status=probed.
    """
    # determine whether target is a file path
    p = Path(target)
    if p.exists():
        summary = probe_document(str(p), n_samples=samples)
        click.echo(json.dumps(summary, indent=2))
        return

    # otherwise assume it's a document_id and load from DB
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == target).one_or_none()
    if not doc:
        click.echo(f"Document {target} not found in DB")
        return
    if not doc.file_path:
        click.echo(f"Document {target} has no file_path recorded")
        return
    summary = probe_document(doc.file_path, n_samples=samples)
    # persist summary
    doc.probe_summary = summary
    doc.ingestion_status = IngestionStatus.probed
    session.add(doc)
    session.commit()
    click.echo(f"Probed document {target}; complexity={summary.get('complexity_score')}")


if __name__ == '__main__':
    cli()
