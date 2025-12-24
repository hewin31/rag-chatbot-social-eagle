"""Adaptive chunking strategies for different block types."""

import re
from typing import List, Dict, Tuple
import uuid


class ChunkResult:
    """Result of chunking a block."""

    def __init__(
        self,
        chunk_id: str,
        block_id: int,
        document_id: str,
        page_number: int,
        content_type: str,
        chunk_text: str,
        token_count: int,
        overlap_with_prev: bool,
        confidence_score: int,
        creation_method: str,
    ):
        self.chunk_id = chunk_id
        self.block_id = block_id
        self.document_id = document_id
        self.page_number = page_number
        self.content_type = content_type
        self.chunk_text = chunk_text
        self.token_count = token_count
        self.overlap_with_prev = overlap_with_prev
        self.confidence_score = confidence_score
        self.creation_method = creation_method


def estimate_tokens(text: str) -> int:
    """Rough token count: split by whitespace."""
    return len(text.split())


def chunk_text_semantic(
    block_id: int,
    document_id: str,
    page_number: int,
    content: str,
    confidence: int,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[ChunkResult]:
    """Split text block into semantic chunks by paragraph.

    Preserves paragraph boundaries, applies overlap for context.
    """
    chunks = []

    # split by paragraph (double newline or single newline)
    paragraphs = re.split(r'\n\s*\n|\n', content.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        # fallback: create single chunk
        chunk = ChunkResult(
            chunk_id=str(uuid.uuid4()),
            block_id=block_id,
            document_id=document_id,
            page_number=page_number,
            content_type="text",
            chunk_text=content,
            token_count=estimate_tokens(content),
            overlap_with_prev=False,
            confidence_score=confidence,
            creation_method="semantic_fallback",
        )
        chunks.append(chunk)
        return chunks

    current_chunk = []
    current_tokens = 0
    prev_chunk_text = None

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # if adding this para exceeds limit, finalize current chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            overlap = False
            if prev_chunk_text and overlap_tokens > 0:
                # add overlap context from previous chunk
                overlap_context = " ".join(prev_chunk_text.split()[-overlap_tokens:])
                chunk_text = overlap_context + "\n\n" + chunk_text
                overlap = True

            chunk = ChunkResult(
                chunk_id=str(uuid.uuid4()),
                block_id=block_id,
                document_id=document_id,
                page_number=page_number,
                content_type="text",
                chunk_text=chunk_text,
                token_count=estimate_tokens(chunk_text),
                overlap_with_prev=overlap,
                confidence_score=confidence,
                creation_method="semantic_paragraph",
            )
            chunks.append(chunk)
            prev_chunk_text = chunk_text
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # finalize last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        overlap = False
        if prev_chunk_text and overlap_tokens > 0:
            overlap_context = " ".join(prev_chunk_text.split()[-overlap_tokens:])
            chunk_text = overlap_context + "\n\n" + chunk_text
            overlap = True

        chunk = ChunkResult(
            chunk_id=str(uuid.uuid4()),
            block_id=block_id,
            document_id=document_id,
            page_number=page_number,
            content_type="text",
            chunk_text=chunk_text,
            token_count=estimate_tokens(chunk_text),
            overlap_with_prev=overlap,
            confidence_score=confidence,
            creation_method="semantic_paragraph",
        )
        chunks.append(chunk)

    return chunks


def chunk_table(
    block_id: int,
    document_id: str,
    page_number: int,
    content: str,
    confidence: int,
) -> List[ChunkResult]:
    """Split table block by rows with header preservation.

    Each chunk = header rows + logical row group.
    """
    chunks = []
    rows = content.strip().split('\n')

    if not rows:
        return chunks

    # assume first row is header
    header = rows[0]
    data_rows = rows[1:]

    if not data_rows:
        # just header
        chunk = ChunkResult(
            chunk_id=str(uuid.uuid4()),
            block_id=block_id,
            document_id=document_id,
            page_number=page_number,
            content_type="table",
            chunk_text=content,
            token_count=estimate_tokens(content),
            overlap_with_prev=False,
            confidence_score=confidence,
            creation_method="table_header_only",
        )
        chunks.append(chunk)
        return chunks

    # group rows: max 10 rows per chunk
    rows_per_chunk = 10
    for i in range(0, len(data_rows), rows_per_chunk):
        group = data_rows[i : i + rows_per_chunk]
        chunk_text = header + '\n' + '\n'.join(group)
        chunk = ChunkResult(
            chunk_id=str(uuid.uuid4()),
            block_id=block_id,
            document_id=document_id,
            page_number=page_number,
            content_type="table",
            chunk_text=chunk_text,
            token_count=estimate_tokens(chunk_text),
            overlap_with_prev=(i > 0),  # overlap if not first group
            confidence_score=confidence,
            creation_method="table_rowwise",
        )
        chunks.append(chunk)

    return chunks


def chunk_image(
    block_id: int,
    document_id: str,
    page_number: int,
    content: str,
    confidence: int,
) -> List[ChunkResult]:
    """Single chunk for image block (with caption/OCR if present)."""
    chunk = ChunkResult(
        chunk_id=str(uuid.uuid4()),
        block_id=block_id,
        document_id=document_id,
        page_number=page_number,
        content_type="image",
        chunk_text=content,
        token_count=estimate_tokens(content),
        overlap_with_prev=False,
        confidence_score=confidence,
        creation_method="image_single",
    )
    return [chunk]


def adaptive_chunk(
    block_id: int,
    document_id: str,
    page_number: int,
    block_type: str,
    content: str,
    confidence: int,
) -> List[ChunkResult]:
    """Dispatch to appropriate chunking strategy based on block type."""
    if block_type == "text":
        return chunk_text_semantic(block_id, document_id, page_number, content, confidence)
    elif block_type.startswith("table"):
        return chunk_table(block_id, document_id, page_number, content, confidence)
    elif block_type == "image":
        return chunk_image(block_id, document_id, page_number, content, confidence)
    else:
        # fallback: treat as text
        return chunk_text_semantic(block_id, document_id, page_number, content, confidence)
