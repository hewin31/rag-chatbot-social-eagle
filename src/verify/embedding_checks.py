from src.db import get_session
from src.db.models import Embedding, Chunk


def embedding_count_check(document_id: str) -> dict:
    with get_session() as session:
        chunk_count = session.query(Chunk).filter(Chunk.document_id == document_id, Chunk.confidence_score != None).count()
        emb_count = session.query(Embedding).filter(Embedding.document_id == document_id).count()
        issues = []
        if emb_count == 0:
            issues.append("no_embeddings_found")
        if emb_count > chunk_count:
            issues.append("more_embeddings_than_chunks")
        return {"document_id": document_id, "chunks": chunk_count, "embeddings": emb_count, "issues": issues}


def sample_retrieval_check(document_id: str, sample_queries: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict:
    from src.embeddings.embeddings import query_document_index
    results = {}
    for q in sample_queries:
        res = query_document_index(document_id, q, top_k=3, model_name=model_name)
        results[q] = res
    return {"document_id": document_id, "samples": results}


def traceability_check(document_id: str) -> dict:
    with get_session() as session:
        emb_records = session.query(Embedding).filter(Embedding.document_id == document_id).all()
        missing_chunks = []
        for e in emb_records:
            c = session.query(Chunk).filter(Chunk.chunk_id == e.chunk_id).first()
            if not c:
                missing_chunks.append(str(e.chunk_id))
        return {"document_id": document_id, "embeddings": len(emb_records), "missing_chunk_refs": missing_chunks}
