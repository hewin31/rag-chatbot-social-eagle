"""Cross-validation: compare extracted blocks with source PDF."""

import random
from ..db import get_session
from ..db.models import Document, Block
from ..ingest.parsing import extract_text_from_page


def cross_check_document(document_id, sample_size=3):
    """Sample blocks from DB and compare with fresh extraction from PDF.

    Returns validation report with pass/fail and variance info.
    """
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == document_id).one_or_none()
    if not doc:
        return {"error": "document not found"}

    # get all text blocks
    blocks = session.query(Block).filter(
        Block.document_id == document_id,
        Block.block_type == "text"
    ).all()

    if not blocks:
        return {"error": "no text blocks found to validate"}

    sample_blocks = random.sample(blocks, min(sample_size, len(blocks)))
    validations = []

    for block in sample_blocks:
        # re-extract from PDF
        try:
            fresh = extract_text_from_page(doc.file_path, block.page_number)
            fresh_len = len(fresh.content.strip())
        except Exception as e:
            validations.append({
                "page_number": block.page_number,
                "status": "error",
                "error": str(e),
            })
            continue

        db_len = len(block.content.strip())

        # allow 10% variance
        variance = abs(fresh_len - db_len) / max(1, db_len)
        threshold = 0.1
        status = "pass" if variance <= threshold else "fail"

        validations.append({
            "page_number": block.page_number,
            "db_content_length": db_len,
            "fresh_content_length": fresh_len,
            "variance": round(variance, 3),
            "status": status,
        })

    issues = [v for v in validations if v["status"] != "pass"]
    report = {
        "document_id": str(document_id),
        "sample_size": len(sample_blocks),
        "validations": validations,
        "issues": issues,
        "overall": "pass" if not issues else "fail",
    }
    return report
