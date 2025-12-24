from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Enum, BigInteger, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
import datetime
import uuid

Base = declarative_base()

class IngestionStatus(str, enum.Enum):
    received = "received"
    probed = "probed"
    parsed = "parsed"
    verified = "verified"
    failed = "failed"

class Document(Base):
    __tablename__ = "documents"
    document_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(Text, nullable=False)
    page_count = Column(Integer, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    source = Column(String, nullable=True)
    ingestion_status = Column(Enum(IngestionStatus), default=IngestionStatus.received)
    probe_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Page(Base):
    __tablename__ = "pages"
    id = Column(Integer, primary_key=True)
    document_id = Column(UUID(as_uuid=True))
    page_number = Column(Integer)
    page_summary = Column(JSON, nullable=True)

class Block(Base):
    __tablename__ = "blocks"
    id = Column(Integer, primary_key=True)
    document_id = Column(UUID(as_uuid=True))
    page_number = Column(Integer)
    block_type = Column(String)
    content = Column(Text)
    extraction_method = Column(String)
    confidence = Column(Integer)

class Chunk(Base):
    __tablename__ = "chunks"
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    block_id = Column(Integer, nullable=False)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    page_number = Column(Integer, nullable=False)
    content_type = Column(String)  # text, table, image, diagram
    chunk_text = Column(Text)
    token_count = Column(Integer, nullable=True)
    overlap_with_prev = Column(Boolean, default=False)
    confidence_score = Column(Integer)  # propagated from block
    creation_method = Column(String)  # adaptive, semantic, table_based, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), nullable=False)
    block_id = Column(Integer, nullable=True)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    vector_index = Column(Integer, nullable=False)  # position in FAISS index
    vector_dim = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    index_path = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
