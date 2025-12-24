import click
from .parsing import parse_document, classify_page_type
from ..db import get_session
from ..db.models import Document, Block, Page, IngestionStatus
import uuid


@click.group()
def cli():
    pass


@cli.command()
@click.argument('target')
def parse(target):
    """Parse a document (text/tables) and populate blocks table.

    If `target` is a document_id in the DB, loads the PDF path and parses.
    Populates the `blocks` table with extracted content, method, and confidence.
    Sets ingestion_status=parsed when complete.
    """
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == target).one_or_none()
    if not doc:
        click.echo(f"Document {target} not found in DB")
        return

    pdf_path = doc.file_path
    click.echo(f"Parsing document {target} from {pdf_path}")

    # parse all pages
    try:
        per_page_blocks = parse_document(pdf_path)
    except Exception as e:
        click.echo(f"Error parsing document: {e}")
        return

    # insert blocks into DB
    total_blocks = 0
    for page_num, blocks in per_page_blocks.items():
        page_type = classify_page_type(pdf_path, page_num)

        for block_result in blocks:
            block = Block(
                document_id=doc.document_id,
                page_number=page_num,
                block_type=block_result.block_type,
                content=block_result.content,
                extraction_method=block_result.extraction_method,
                confidence=block_result.confidence,
            )
            session.add(block)
            total_blocks += 1

    # update document status
    doc.ingestion_status = IngestionStatus.parsed
    session.add(doc)
    session.commit()

    click.echo(
        f"Parsed document {target}; inserted {total_blocks} blocks; status=parsed"
    )


if __name__ == '__main__':
    cli()
