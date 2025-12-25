import logging
import time
import uuid
from typing import List, Dict, Any
from sqlalchemy import select, or_
import spacy

from ..db import get_session
from ..db.models import Chunk, Entity, Relationship, QueryLog

# Optional: Import Vector DB libraries if available
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    VECTOR_DEPS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self):
        self.session = get_session()
        self._load_spacy()
        self._load_vector_resources()

    def _load_spacy(self):
        """Loads the spaCy model for query entity extraction."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _load_vector_resources(self):
        """Loads FAISS index and Embedding model if available."""
        self.vector_model = None
        self.faiss_index = None
        
        if VECTOR_DEPS_AVAILABLE:
            # TODO: Load actual index path from config or DB
            # self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
            # self.faiss_index = faiss.read_index("path/to/index.faiss")
            pass
        else:
            logger.warning("Vector dependencies not found. Running in SQL-fallback mode.")

    def retrieve(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main entry point for RAG retrieval.
        """
        start_time = time.time()
        logger.info(f"Retrieving for query: '{query_text}'")

        # 1. Classify Query (Heuristic)
        query_type = self._classify_query(query_text)
        
        # 2. Vector Search (Semantic Context)
        vector_results = self._vector_search(query_text, top_k)
        
        # 3. KG Search (Relational Context)
        kg_results = self._kg_search(query_text)
        
        # 4. Consolidate Results
        final_results = {
            "query": query_text,
            "type": query_type,
            "chunks": vector_results,
            "graph": kg_results
        }
        
        # 5. Log for Provenance
        duration = int((time.time() - start_time) * 1000)
        self._log_query(query_text, query_type, vector_results, kg_results, duration)
        
        return final_results

    def _classify_query(self, query: str) -> str:
        """Determines if query is relational or semantic."""
        relational_triggers = ["who", "relationship", "connect", "between", "how is", "related", "what is the link"]
        if any(t in query.lower() for t in relational_triggers):
            return "relational"
        return "semantic"

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Performs semantic search. Falls back to SQL ILIKE if no vector index is loaded.
        """
        if self.faiss_index and self.vector_model:
            # Real Vector Search Implementation
            # query_vec = self.vector_model.encode([query])
            # D, I = self.faiss_index.search(query_vec, top_k)
            # Map I (indices) back to Chunk IDs via Embedding table
            return []
        else:
            # Fallback: Simple Keyword Search in SQL
            logger.info("Using SQL keyword fallback for vector search.")
            stmt = select(Chunk).filter(Chunk.chunk_text.ilike(f"%{query}%")).limit(top_k)
            chunks = self.session.execute(stmt).scalars().all()
            
            return [
                {
                    "chunk_id": str(c.chunk_id),
                    "text": c.chunk_text[:200] + "...", # Preview
                    "score": 0.0, # No similarity score in keyword search
                    "source": "sql_fallback"
                }
                for c in chunks
            ]

    def _kg_search(self, query: str) -> Dict[str, Any]:
        """
        Extracts entities from query and finds 1-hop neighbors in the graph.
        """
        doc = self.nlp(query)
        # Extract entities from the user's question
        query_entities = [ent.text for ent in doc.ents]
        
        if not query_entities:
            return {"entities": [], "relationships": []}
            
        found_entities = []
        found_relationships = []
        
        for ent_text in query_entities:
            # Find this entity in our DB (fuzzy match)
            stmt = select(Entity).filter(Entity.entity_text.ilike(f"%{ent_text}%"))
            db_entities = self.session.execute(stmt).scalars().all()
            
            for db_ent in db_entities:
                found_entities.append({"name": db_ent.entity_text, "type": db_ent.entity_type})
                
                # Find relationships where this entity is Source or Target
                rel_stmt = select(Relationship).filter(
                    or_(
                        Relationship.source_entity_id == db_ent.entity_id,
                        Relationship.target_entity_id == db_ent.entity_id
                    )
                )
                rels = self.session.execute(rel_stmt).scalars().all()
                
                for r in rels:
                    found_relationships.append({
                        "source_id": str(r.source_entity_id),
                        "target_id": str(r.target_entity_id),
                        "type": r.relationship_type
                    })
                    
        return {"entities": found_entities, "relationships": found_relationships}

    def _log_query(self, text, q_type, chunks, graph, duration):
        """Saves query execution details to DB."""
        try:
            chunk_ids = [c['chunk_id'] for c in chunks]
            log = QueryLog(
                query_text=text,
                query_type=q_type,
                retrieved_chunk_ids=chunk_ids,
                retrieved_graph_data=graph,
                execution_time_ms=duration
            )
            self.session.add(log)
            self.session.commit()
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            self.session.rollback()

if __name__ == "__main__":
    # Test the engine
    engine = RetrievalEngine()
    result = engine.retrieve("What is the relationship between Python and RAG?")
    import json
    print(json.dumps(result, indent=2, default=str))