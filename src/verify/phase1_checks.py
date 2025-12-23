"""Simple verification scripts for Phase 1 (manual checks).

These helpers produce queries or simple checks to confirm a document was registered
and metadata matches the stored file.
"""
from ..db import get_session
from ..db.models import Document
from pathlib import Path

def check_document_exists(document_id):
    session = get_session()
    doc = session.query(Document).filter(Document.document_id == document_id).one_or_none()
    return doc

def verify_file_matches_metadata(document_id):
    doc = check_document_exists(document_id)
    if not doc:
        return False, 'document not found'
    path = Path(doc.file_path)
    if not path.exists():
        return False, 'file missing'
    size = path.stat().st_size
    if doc.file_size_bytes != size:
        return False, f'size mismatch db={doc.file_size_bytes} fs={size}'
    return True, 'ok'
