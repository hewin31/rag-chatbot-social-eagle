import pdfplumber
from pathlib import Path

def extract_basic_metadata(pdf_path: str) -> dict:
    path = Path(pdf_path)
    result = {"file_size_bytes": path.stat().st_size, "page_count": None}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["page_count"] = len(pdf.pages)
    except Exception:
        result["page_count"] = None
    return result
