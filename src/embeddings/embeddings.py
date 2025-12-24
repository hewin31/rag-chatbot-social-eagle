import os
import json
import math
import tempfile
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import faiss

from sqlalchemy.orm import Session

from src.db import get_session
from src.db.models import Chunk, Embedding

# optional imports
try:
    import openai
except Exception:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def _batch(iterable, n=64):
    it = iter(iterable)
    while True:
        batch = []
        try:
            for _ in range(n):
                batch.append(next(it))
        except StopIteration:
            if batch:
                yield batch
            break
        yield batch


def get_embedding_model(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a callable that maps list[str] -> np.ndarray

    Supports OpenAI 'openai:text-embedding-3-small' or a SentenceTransformers model name.
    """
    if name.startswith("openai:"):
        if openai is None:
            raise RuntimeError("openai package not available; install openai in requirements.txt")
        model_name = name.split("openai:", 1)[1]

        def _openai_embed(texts: List[str]) -> np.ndarray:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            openai.api_key = key
            # OpenAI API: chunk call
            resp = openai.Embedding.create(model=model_name, input=texts)
            vecs = [r["embedding"] for r in resp["data"]]
            return np.array(vecs, dtype="float32")

        return _openai_embed

    # fallback to sentence-transformers
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    model = SentenceTransformer(name)

    def _st_embed(texts: List[str]) -> np.ndarray:
        arr = model.encode(texts, show_progress_bar=False)
        return np.array(arr, dtype="float32")

    return _st_embed


def _ensure_faiss_index(path: Path, dim: int) -> Tuple[faiss.IndexFlatIP, int]:
    """Load or create a FAISS index (inner-product normalized for cosine similarity).

    Returns (index, current_count)
    """
    if path.exists():
        index = faiss.read_index(str(path))
        current_n = index.ntotal
        return index, current_n
    else:
        # Using normalized vectors + IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(dim)
        return index, 0


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def generate_embeddings_for_document(document_id: str,
                                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                     confidence_threshold: float = 0.85,
                                     batch_size: int = 64,
                                     index_dir: Optional[str] = None) -> Dict:
    """Generate embeddings for trusted chunks of a document.

    - Filters chunks by `confidence_score >= confidence_threshold`.
    - Generates vectors in batches and stores them in a FAISS index file under `data/embeddings/{document_id}.index`.
    - Records mapping rows in the `embeddings` table.

    Returns a summary dict with counts and paths.
    """
    idx_dir = EMBEDDINGS_DIR if index_dir is None else Path(index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    index_path = idx_dir / f"{document_id}.index"

    with get_session() as session:
        # fetch eligible chunks
        chunks = session.query(Chunk).filter(
            Chunk.document_id == document_id,
            Chunk.confidence_score >= int(confidence_threshold * 100) if isinstance(Chunk.confidence_score.type, object) else Chunk.confidence_score >= int(confidence_threshold * 100)
        ).all()

        # Fallback more tolerant filter if no results (handle 0-1 floats vs 0-100 ints)
        if not chunks:
            chunks = session.query(Chunk).filter(
                Chunk.document_id == document_id,
                Chunk.confidence_score >= int(confidence_threshold * 100)
            ).all()

        texts = [c.chunk_text or "" for c in chunks]
        if not texts:
            return {"document_id": document_id, "embeddings_created": 0, "index_path": str(index_path)}

        embed_fn = get_embedding_model(model_name)

        # compute one example vector to determine dim
        sample_vecs = embed_fn([texts[0]])
        dim = sample_vecs.shape[1]

        # prepare or load index
        index, current_n = _ensure_faiss_index(index_path, dim)

        # if using IP index for cosine, we must normalize
        created = 0
        vec_index = current_n
        for batch_idxs in _batch(list(range(len(texts))), batch_size):
            batch_texts = [texts[i] for i in batch_idxs]
            vecs = embed_fn(batch_texts)
            vecs = _normalize_vectors(vecs)
            index.add(vecs)

            # persist mapping records
            for i, local_idx in enumerate(batch_idxs):
                chunk = chunks[local_idx]
                emb = Embedding(
                    chunk_id=chunk.chunk_id,
                    block_id=chunk.block_id,
                    document_id=chunk.document_id,
                    vector_index=vec_index,
                    vector_dim=dim,
                    model_name=model_name,
                    index_path=str(index_path),
                    metadata_json={"content_type": chunk.content_type, "token_count": chunk.token_count},
                )
                session.add(emb)
                vec_index += 1
                created += 1
            session.commit()

        # write index to disk
        faiss.write_index(index, str(index_path))

    return {"document_id": document_id, "embeddings_created": created, "index_path": str(index_path), "vector_dim": dim}


def load_index_for_document(document_id: str, index_dir: Optional[str] = None) -> Tuple[faiss.IndexFlatIP, Path]:
    idx_dir = EMBEDDINGS_DIR if index_dir is None else Path(index_dir)
    index_path = idx_dir / f"{document_id}.index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    return index, index_path


def query_document_index(document_id: str, query: str, top_k: int = 5, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Tuple[int, float]]:
    """Return list of (vector_index, score) for top_k matches for a query string.
    Score is cosine similarity in [0,1].
    """
    index, _ = load_index_for_document(document_id)
    embed_fn = get_embedding_model(model_name)
    qv = embed_fn([query])
    qv = _normalize_vectors(qv)
    D, I = index.search(qv, top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))
