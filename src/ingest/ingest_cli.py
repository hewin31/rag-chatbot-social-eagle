import click
from .storage import store_pdf
from .metadata import extract_basic_metadata
from ..db import get_session
from ..db.models import Document, IngestionStatus
from ..config import ensure_paths
import os


@click.group()
def cli():
    ensure_paths()


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
def ingest(pdf_path):
    """Ingest a PDF into the document registry (Phase 1).

    This command will copy the file into local storage, extract basic metadata,
    and create a `documents` record with status `received`.
    """
    filename = os.path.basename(pdf_path)
    dest = store_pdf(pdf_path, filename)
    meta = extract_basic_metadata(str(dest))

    session = get_session()
    doc = Document(
        filename=filename,
        file_path=str(dest),
        page_count=meta.get('page_count'),
        file_size_bytes=meta.get('file_size_bytes'),
        ingestion_status=IngestionStatus.received,
    )
    session.add(doc)
    session.commit()
    print(f"Ingested document_id={doc.document_id} path={doc.file_path}")


if __name__ == '__main__':
    cli()
