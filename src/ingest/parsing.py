"""Deterministic parsing engine for text and table extraction."""

import pdfplumber
import re
from typing import List, Dict


class ExtractionResult:
    """Result of extracting a block from a page."""

    def __init__(
        self,
        block_type: str,
        content: str,
        extraction_method: str,
        confidence: int,
        page_number: int,
    ):
        self.block_type = block_type
        self.content = content
        self.extraction_method = extraction_method
        self.confidence = confidence
        self.page_number = page_number


def extract_text_from_page(pdf_path: str, page_number: int) -> ExtractionResult:
    """Extract text from a page using pdfplumber text extraction.

    Confidence based on text length and validity.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 0 or page_number >= len(pdf.pages):
            raise IndexError("page_number out of range")
        page = pdf.pages[page_number]
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = ""
            confidence = 10
            return ExtractionResult(
                block_type="text",
                content=f"[extraction error: {e}]",
                extraction_method="pdfplumber_text_error",
                confidence=confidence,
                page_number=page_number,
            )

    # confidence heuristic
    text_len = len(text.strip())
    if text_len == 0:
        confidence = 20
    elif text_len < 100:
        confidence = 60
    elif text_len < 1000:
        confidence = 80
    else:
        confidence = 95

    return ExtractionResult(
        block_type="text",
        content=text,
        extraction_method="pdfplumber_text",
        confidence=confidence,
        page_number=page_number,
    )


def extract_tables_from_page(pdf_path: str, page_number: int) -> List[ExtractionResult]:
    """Extract tables from a page using pdfplumber table detection.

    Returns a list of ExtractionResult objects, one per table.
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 0 or page_number >= len(pdf.pages):
            raise IndexError("page_number out of range")
        page = pdf.pages[page_number]
        try:
            tables = page.find_tables()
        except Exception:
            tables = []

    if not tables:
        return results

    for i, table in enumerate(tables):
        try:
            # Convert table to CSV-like string
            rows = table.rows
            csv_content = "\n".join(
                ",".join(str(cell) if cell else "" for cell in row) for row in rows
            )
        except Exception as e:
            csv_content = f"[table extraction error: {e}]"
            confidence = 20
        else:
            # confidence based on number of rows/columns
            num_rows = len(table.rows) if hasattr(table, "rows") else 0
            num_cols = len(table.rows[0]) if table.rows else 0
            if num_rows < 3 or num_cols < 2:
                confidence = 60
            else:
                confidence = 85

        results.append(
            ExtractionResult(
                block_type=f"table_{i}",
                content=csv_content,
                extraction_method="pdfplumber_table",
                confidence=confidence,
                page_number=page_number,
            )
        )

    return results


def parse_document(pdf_path: str) -> Dict[int, List[ExtractionResult]]:
    """Parse all pages in a document.

    Returns dict: {page_number: [ExtractionResult, ...]}
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    results = {}
    for page_num in range(total_pages):
        page_blocks = []
        try:
            # always extract text
            text_result = extract_text_from_page(pdf_path, page_num)
            page_blocks.append(text_result)
        except Exception:
            pass

        try:
            # extract tables if present
            table_results = extract_tables_from_page(pdf_path, page_num)
            page_blocks.extend(table_results)
        except Exception:
            pass

        results[page_num] = page_blocks

    return results


def classify_page_type(pdf_path: str, page_number: int) -> str:
    """Classify page based on content.

    Returns one of: text_heavy, table_heavy, image_heavy, mixed, unknown
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 0 or page_number >= len(pdf.pages):
            raise IndexError("page_number out of range")
        page = pdf.pages[page_number]
        try:
            text = page.extract_text() or ""
            text_len = len(text.strip())
        except Exception:
            text_len = 0

        try:
            tables = page.find_tables()
            table_count = len(tables) if tables else 0
        except Exception:
            table_count = 0

        try:
            images = page.images or []
            image_count = len(images)
        except Exception:
            image_count = 0

    if text_len == 0 and image_count > 0:
        return "image_heavy"
    if table_count >= 2:
        return "table_heavy"
    if text_len > 500 and table_count == 0 and image_count == 0:
        return "text_heavy"
    if text_len > 0 or table_count > 0 or image_count > 0:
        return "mixed"
    return "unknown"
