"""RAG Retrieval Engine: Handles hybrid vector + graph search."""
import logging
import time
import uuid
from typing import List, Dict, Any, Tuple
from sqlalchemy import select, or_
import spacy

from .session import get_session
from .models import Chunk, Entity, Relationship, QueryLog, Embedding

# Optional: Import Vector DB libraries if available
VECTOR_SEARCH_ERROR = None
try:
    from src.embeddings.embeddings import query_document_index
    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    VECTOR_SEARCH_AVAILABLE = False
    VECTOR_SEARCH_ERROR = str(e)
except Exception as e:
    VECTOR_SEARCH_AVAILABLE = False
    VECTOR_SEARCH_ERROR = f"Unexpected error: {e}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self):
        self.session = get_session()
        self._load_spacy()

    def _load_spacy(self):
        """Loads the spaCy model for query entity extraction."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def retrieve(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main entry point for RAG retrieval.
        """
        start_time = time.time()
        
        # Collect debug logs to return with response instead of printing immediately
        debug_logs = [f"Retrieving for query: '{query_text}'"]

        # 1. Classify Query (Heuristic)
        query_type = self._classify_query(query_text)
        
        # 2. Vector Search (Semantic Context)
        vector_results, vector_logs = self._vector_search(query_text, top_k)
        debug_logs.extend(vector_logs)
        
        # 3. KG Search (Relational Context)
        try:
            kg_results, kg_logs = self._kg_search(query_text)
            debug_logs.extend(kg_logs)
        except Exception as e:
            debug_logs.append(f"ERROR: KG search failed: {e}")
            kg_results = {"entities": [], "relationships": []}
        
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
        
        # 6. Attach Execution Stats (for post-processing display)
        final_results["execution_stats"] = {
            "duration_ms": duration,
            "debug_logs": debug_logs,
            "metrics": {
                "chunks": len(vector_results),
                "entities": len(kg_results.get("entities", [])),
                "relationships": len(kg_results.get("relationships", []))
            }
        }
        
        return final_results

    def _classify_query(self, query: str) -> str:
        """Determines if query is relational or semantic."""
        relational_triggers = ["who", "relationship", "connect", "between", "how is", "related", "what is the link"]
        if any(t in query.lower() for t in relational_triggers):
            return "relational"
        return "semantic"

    def _vector_search(self, query: str, top_k: int) -> Tuple[List[Dict], List[str]]:
        """
        Performs semantic search. Falls back to SQL ILIKE if no vector index is loaded.
        """
        logs = []
        logs.append(f"DEBUG: VECTOR_SEARCH_AVAILABLE = {VECTOR_SEARCH_AVAILABLE}")
        if not VECTOR_SEARCH_AVAILABLE and VECTOR_SEARCH_ERROR:
            logs.append(f"DEBUG: Vector search import failed: {VECTOR_SEARCH_ERROR}")

        if VECTOR_SEARCH_AVAILABLE:
            try:
                # 1. Find all documents that have embeddings
                stmt = select(Embedding.document_id).distinct()
                doc_ids = self.session.execute(stmt).scalars().all()
                logs.append(f"DEBUG: Found {len(doc_ids)} documents with embeddings: {doc_ids}")
                if not doc_ids:
                    logs.append("DEBUG: No embeddings found in DB. Have you run the ingestion script?")
                
                candidates = [] # List of (score, doc_id, vector_index)
                
                # 2. Federated Search across all document indices
                for doc_id in doc_ids:
                    try:
                        # Returns list of (vector_index, score)
                        results = query_document_index(str(doc_id), query, top_k=top_k)
                        logs.append(f"DEBUG: Document {doc_id} returned {len(results)} matches.")
                        for vec_idx, score in results:
                            candidates.append((score, doc_id, vec_idx))
                    except Exception as e:
                        # Index might be missing or other error
                        logs.append(f"DEBUG: Failed to query index for doc {doc_id}: {e}")
                        continue
                
                # 3. Sort globally by score and take top_k
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = candidates[:top_k]
                logs.append(f"DEBUG: Top {len(top_candidates)} global candidates selected.")
                
                # 4. Resolve to Chunks
                final_results = []
                for score, doc_id, vec_idx in top_candidates:
                    # Find the chunk_id mapping
                    emb_stmt = select(Embedding).filter_by(document_id=doc_id, vector_index=vec_idx)
                    embedding_record = self.session.execute(emb_stmt).scalars().first()
                    
                    if embedding_record:
                        chunk = self.session.execute(select(Chunk).filter_by(chunk_id=embedding_record.chunk_id)).scalars().first()
                        if chunk:
                            final_results.append({
                                "chunk_id": str(chunk.chunk_id),
                                "text": chunk.chunk_text,
                                "score": float(score),
                                "source": "vector_search"
                            })
                
                if final_results:
                    logs.append(f"DEBUG: Vector search successful. Returning {len(final_results)} results.")
                    return final_results, logs
                else:
                    logs.append("DEBUG: Vector search finished but resolved to 0 chunks.")
                    
            except Exception as e:
                logs.append(f"ERROR: Vector search failed: {e}")
                # Fallback to SQL below
        
        # Fallback: Simple Keyword Search in SQL
        if not VECTOR_SEARCH_AVAILABLE:
             logs.append("Vector search unavailable. Using SQL keyword fallback.")
        else:
             logs.append("Vector search yielded no results or failed. Using SQL keyword fallback.")
             
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
        ], logs

    def _kg_search(self, query: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts entities from query and finds 1-hop neighbors in the graph.
        """
        logs = []
        doc = self.nlp(query)
        # Extract entities from the user's question
        query_entities = [ent.text for ent in doc.ents]
        
        # Fallback: If no named entities found, try to find important nouns
        if not query_entities:
            # Filter for Nouns and Proper Nouns, excluding stop words and generic terms
            ignored_terms = {"relationship", "link", "connection", "between", "what", "how"}
            query_entities = [
                token.text for token in doc 
                if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in ignored_terms
            ]
            logs.append(f"DEBUG: NER failed. Fallback to Nouns: {query_entities}")
        else:
            logs.append(f"DEBUG: KG Query Entities Found: {query_entities}")
            
        if not query_entities:
            return {"entities": [], "relationships": []}, logs
            
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
                    
        return {"entities": found_entities, "relationships": found_relationships}, logs

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
    # Using a question with entities known to exist in the DB (e.g., 'Income Tax')
    result = engine.retrieve("What is the relationship between Income Tax and India?")
    
    # Separate stats from the main result for cleaner printing
    stats = result.pop("execution_stats", None)
    
    import json
    print(json.dumps(result, indent=2, default=str))
    
    if stats:
        print("\n" + "="*40)
        print("       DEBUG LOGS & METRICS")
        print("="*40)
        for line in stats.get("debug_logs", []):
            print(line)
        
        m = stats.get("metrics", {})
        print("-" * 40)
        print(f"METRICS: Time={stats.get('duration_ms')}ms | Chunks={m.get('chunks')} | Entities={m.get('entities')} | Relations={m.get('relationships')}")
        print("="*40)