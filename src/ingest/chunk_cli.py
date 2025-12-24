"""CLI for adaptive chunking: convert blocks to chunks."""

import click
from .chunking import adaptive_chunk
from ..db import get_session
from ..db.models import Block, Chunk, Document


@click.group()
def cli():
    pass


@cli.command()
@click.argument('document_id')
def chunk(document_id):
    """Adaptively chunk all blocks for a document.

    Applies different chunking strategies based on block type:
    - text blocks → semantic paragraph-based chunks
    - table blocks → row-wise chunks with header preservation
    - image blocks → single chunks with caption/OCR
    """
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == document_id).one_or_none()
    if not doc:
        click.echo(f"Document {document_id} not found")
        return

    # get all blocks for this document
    blocks = session.query(Block).filter(Block.document_id == document_id).all()
    if not blocks:
        click.echo(f"No blocks found for document {document_id}")
        return

    total_chunks = 0
    for block in blocks:
        chunk_results = adaptive_chunk(
            block_id=block.id,
            document_id=document_id,
            page_number=block.page_number,
            block_type=block.block_type,
            content=block.content,
            confidence=block.confidence,
        )

        for chunk_result in chunk_results:
            chunk_obj = Chunk(
                chunk_id=chunk_result.chunk_id,
                block_id=chunk_result.block_id,
                document_id=chunk_result.document_id,
                page_number=chunk_result.page_number,
                content_type=chunk_result.content_type,
                chunk_text=chunk_result.chunk_text,
                token_count=chunk_result.token_count,
                overlap_with_prev=chunk_result.overlap_with_prev,
                confidence_score=chunk_result.confidence_score,
                creation_method=chunk_result.creation_method,
            )
            session.add(chunk_obj)
            total_chunks += 1

    session.commit()
    click.echo(f"Chunked document {document_id}; created {total_chunks} chunks")


if __name__ == '__main__':
    cli()
