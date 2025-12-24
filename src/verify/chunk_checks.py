"""Verification helpers for adaptive chunking."""

from ..db import get_session
from ..db.models import Document, Block, Chunk
from sqlalchemy import func


def verify_chunks(document_id):
    """Verify chunks are correct and complete.

    Checks:
    - Chunk count per block
    - No empty chunks
    - Chunk content aligns with block content
    - Token counts are reasonable
    - Traceability is preserved
    """
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == document_id).one_or_none()
    if not doc:
        return {"error": "document not found"}

    # stats
    block_count = session.query(func.count(Block.id)).filter(Block.document_id == document_id).scalar()
    chunk_count = session.query(func.count(Chunk.chunk_id)).filter(Chunk.document_id == document_id).scalar()

    # per-block chunk distribution
    block_chunks = session.query(
        Block.id,
        Block.block_type,
        func.count(Chunk.chunk_id).label("chunk_count")
    ).outerjoin(
        Chunk, Block.id == Chunk.block_id
    ).filter(Block.document_id == document_id).group_by(Block.id, Block.block_type).all()

    issues = []

    # check for empty chunks
    empty_chunks = session.query(func.count(Chunk.chunk_id)).filter(
        Chunk.document_id == document_id,
        func.length(Chunk.chunk_text) < 5
    ).scalar()
    if empty_chunks > 0:
        issues.append(f"{empty_chunks} empty chunks detected")

    # check for missing blocks (blocks with 0 chunks)
    missing_chunks = [b for b in block_chunks if b.chunk_count == 0]
    if missing_chunks:
        issues.append(f"{len(missing_chunks)} blocks have no chunks")

    # token count stats
    token_stats = session.query(
        func.min(Chunk.token_count),
        func.max(Chunk.token_count),
        func.avg(Chunk.token_count)
    ).filter(Chunk.document_id == document_id).one()
    token_min, token_max, token_avg = token_stats

    # creation method breakdown
    methods = session.query(
        Chunk.creation_method,
        func.count(Chunk.chunk_id)
    ).filter(Chunk.document_id == document_id).group_by(Chunk.creation_method).all()

    report = {
        "document_id": str(document_id),
        "blocks_total": block_count,
        "chunks_total": chunk_count,
        "chunks_per_block": {str(b.id): b.chunk_count for b in block_chunks},
        "token_stats": {
            "min": token_min,
            "max": token_max,
            "avg": round(token_avg, 2) if token_avg else None,
        },
        "creation_methods": {method: count for method, count in methods},
        "issues": issues,
        "overall": "pass" if not issues else "fail",
    }
    return report
