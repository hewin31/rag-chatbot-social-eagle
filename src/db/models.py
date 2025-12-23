from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Enum, BigInteger
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
