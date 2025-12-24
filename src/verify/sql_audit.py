"""SQL-based audit and verification for extracted data integrity."""

from ..db import get_session
from ..db.models import Document, Block
from sqlalchemy import func
import uuid


def audit_document(document_id):
    """Run comprehensive audit on a document.

    Returns dict with statistics and any issues found.
    """
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == document_id).one_or_none()
    if not doc:
        return {"error": "document not found"}

    # basic stats
    block_count = session.query(func.count(Block.id)).filter(Block.document_id == document_id).scalar()
    text_blocks = session.query(func.count(Block.id)).filter(
        Block.document_id == document_id,
        Block.block_type == "text"
    ).scalar()
    table_blocks = session.query(func.count(Block.id)).filter(
        Block.document_id == document_id,
        Block.block_type.like("table_%")
    ).scalar()

    # confidence stats
    conf_stats = session.query(
        func.min(Block.confidence),
        func.max(Block.confidence),
        func.avg(Block.confidence)
    ).filter(Block.document_id == document_id).one()
    conf_min, conf_max, conf_avg = conf_stats

    # extraction method breakdown
    methods = session.query(
        Block.extraction_method,
        func.count(Block.id)
    ).filter(Block.document_id == document_id).group_by(Block.extraction_method).all()

    issues = []

    # check for gaps in page numbers
    page_numbers = session.query(func.distinct(Block.page_number)).filter(
        Block.document_id == document_id
    ).order_by(Block.page_number).all()
    page_numbers = [p[0] for p in page_numbers]
    if page_numbers:
        expected_pages = set(range(min(page_numbers), max(page_numbers) + 1))
        actual_pages = set(page_numbers)
        missing = expected_pages - actual_pages
        if missing:
            issues.append(f"Missing pages: {sorted(missing)}")

    # check for low confidence blocks
    low_conf = session.query(func.count(Block.id)).filter(
        Block.document_id == document_id,
        Block.confidence < 50
    ).scalar()
    if low_conf > 0:
        issues.append(f"{low_conf} blocks with confidence < 50")

    report = {
        "document_id": str(document_id),
        "filename": doc.filename,
        "ingestion_status": doc.ingestion_status.value if doc.ingestion_status else None,
        "page_count": doc.page_count,
        "stats": {
            "total_blocks": block_count,
            "text_blocks": text_blocks,
            "table_blocks": table_blocks,
            "confidence": {
                "min": conf_min,
                "max": conf_max,
                "avg": round(conf_avg, 2) if conf_avg else None,
            },
            "extraction_methods": {method: count for method, count in methods},
        },
        "issues": issues,
    }
    return report


def list_documents_status():
    """List all documents with their ingestion status and block counts."""
    session = get_session()
    docs = session.query(Document).order_by(Document.created_at.desc()).all()
    results = []
    for doc in docs:
        block_count = session.query(func.count(Block.id)).filter(Block.document_id == doc.document_id).scalar()
        results.append({
            "document_id": str(doc.document_id),
            "filename": doc.filename,
            "status": doc.ingestion_status.value if doc.ingestion_status else None,
            "page_count": doc.page_count,
            "blocks_extracted": block_count,
        })
    return results
