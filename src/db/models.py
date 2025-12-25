from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Enum, BigInteger, Boolean, ForeignKey
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
    block_id = Column(Integer, ForeignKey("blocks.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    content_type = Column(String)
    chunk_text = Column(Text)
    token_count = Column(Integer, nullable=True)
    overlap_with_prev = Column(Boolean, default=False)
    confidence_score = Column(Integer)
    creation_method = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.chunk_id"), nullable=False)
    block_id = Column(Integer, nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id"), nullable=False)
    vector_index = Column(Integer, nullable=False)
    vector_dim = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    index_path = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Entity(Base):
    """Knowledge Graph: Entity nodes extracted from chunks."""
    __tablename__ = "entities"
    
    entity_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.chunk_id"), nullable=False)
    block_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    
    entity_text = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    confidence_score = Column(Integer, default=85)
    
    extraction_method = Column(String, default="llm_ner")
    metadata_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class Relationship(Base):
    """Knowledge Graph: Relationships/edges between entities."""
    __tablename__ = "relationships"
    
    relationship_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.document_id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.chunk_id"), nullable=False)
    block_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    
    source_entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.entity_id"), nullable=False)
    target_entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.entity_id"), nullable=False)
    
    relationship_type = Column(String, nullable=False)
    relationship_text = Column(String, nullable=True)
    confidence_score = Column(Integer, default=80)
    
    extraction_method = Column(String, default="llm_relation_extraction")
    metadata_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class QueryLog(Base):
    """Retrieval Layer: Logs user queries and retrieved context for provenance."""
    __tablename__ = "query_logs"
    
    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    query_type = Column(String, nullable=True) # e.g., "semantic", "relational"
    
    retrieved_chunk_ids = Column(JSON, nullable=True) 
    retrieved_graph_data = Column(JSON, nullable=True)
    
    execution_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
