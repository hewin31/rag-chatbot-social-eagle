from pathlib import Path
import shutil
import uuid
from ..config import PDF_STORAGE_PATH

def store_pdf(src_path: str, filename: str) -> Path:
    """Store the PDF under a unique document_id directory and return destination path."""
    doc_id = str(uuid.uuid4())
    dest_dir = PDF_STORAGE_PATH / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(src_path, dest)
    return dest
