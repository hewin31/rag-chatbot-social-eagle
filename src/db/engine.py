"""RAG Retrieval Engine: Handles hybrid vector + graph search."""
import logging
import time
import uuid
from typing import List, Dict, Any, Tuple
from sqlalchemy import select, or_
from collections import defaultdict
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
            
        # 1. Find all matching entities in DB
        all_matches = []
        for ent_text in query_entities:
            stmt = select(Entity).filter(Entity.entity_text.ilike(f"%{ent_text}%"))
            matches = self.session.execute(stmt).scalars().all()
            
            # Prioritize exact matches to reduce noise
            exact_matches = [m for m in matches if m.entity_text.lower() == ent_text.lower()]
            if exact_matches:
                all_matches.extend(exact_matches)
                logs.append(f"DEBUG: Exact match found for '{ent_text}' ({len(exact_matches)} instances).")
            else:
                # Fallback to top 3 shortest partial matches
                matches.sort(key=lambda x: len(x.entity_text))
                top_partial = matches[:3]
                all_matches.extend(top_partial)
                if top_partial:
                    logs.append(f"DEBUG: Partial matches for '{ent_text}': {[e.entity_text for e in top_partial]}")
            
        if not all_matches:
             return {"entities": [], "relationships": []}, logs

        # Deduplicate matches by ID
        unique_entities = {e.entity_id: e for e in all_matches}
        target_entity_ids = list(unique_entities.keys())
        
        logs.append(f"DEBUG: Found {len(target_entity_ids)} matching entities in DB.")

        # 2. Fetch all potential relationships (1-hop)
        rel_stmt = select(Relationship).filter(
            or_(
                Relationship.source_entity_id.in_(target_entity_ids),
                Relationship.target_entity_id.in_(target_entity_ids)
            )
        )
        all_rels = self.session.execute(rel_stmt).scalars().all()
        
        # 3. Apply Relevance Filtering
        direct_rels = []
        expansion_rels = []
        
        # Track connectivity to find "Bridge Nodes" (2-hop connections)
        neighbor_connectivity = defaultdict(set)
        
        for r in all_rels:
            is_source_in = r.source_entity_id in unique_entities
            is_target_in = r.target_entity_id in unique_entities
            
            if is_source_in and is_target_in:
                direct_rels.append(r)
            else:
                expansion_rels.append(r)
                # Identify the neighbor and which query entity it connects to
                if is_source_in:
                    neighbor_id = r.target_entity_id
                    connected_query_id = r.source_entity_id
                else:
                    neighbor_id = r.source_entity_id
                    connected_query_id = r.target_entity_id
                
                neighbor_connectivity[neighbor_id].add(connected_query_id)
        
        # Identify Bridge Nodes: Neighbors connected to > 1 distinct query entity
        bridge_neighbor_ids = {
            nid for nid, connected in neighbor_connectivity.items() 
            if len(connected) > 1
        }
        
        if bridge_neighbor_ids:
            logs.append(f"DEBUG: Found {len(bridge_neighbor_ids)} bridge nodes connecting query entities.")

        # --- Targeted Neighborhood Scanning ---
        # Check if any query terms were missed in the initial DB lookup
        found_texts = {e.entity_text.lower() for e in unique_entities.values()}
        missing_terms = [q.lower() for q in query_entities if q.lower() not in found_texts]
        
        targeted_rels = []
        if missing_terms and expansion_rels:
            logs.append(f"DEBUG: Missing terms {missing_terms}. Scanning neighbors...")
            
            # Pre-fetch neighbor entities to check their text
            neighbor_ids = set()
            for r in expansion_rels:
                if r.source_entity_id in unique_entities:
                    neighbor_ids.add(r.target_entity_id)
                else:
                    neighbor_ids.add(r.source_entity_id)
            
            if neighbor_ids:
                stmt_neighbors = select(Entity).filter(Entity.entity_id.in_(neighbor_ids))
                neighbor_entities = self.session.execute(stmt_neighbors).scalars().all()
                neighbor_map = {e.entity_id: e for e in neighbor_entities}
                
                for r in expansion_rels:
                    nid = r.target_entity_id if r.source_entity_id in unique_entities else r.source_entity_id
                    if nid in neighbor_map:
                        n_text = neighbor_map[nid].entity_text.lower()
                        # Fuzzy match: check if missing term is part of neighbor text
                        if any(term in n_text for term in missing_terms):
                            targeted_rels.append(r)
                            unique_entities[nid] = neighbor_map[nid] # Optimization

            if targeted_rels:
                logs.append(f"DEBUG: Found {len(targeted_rels)} relationships connecting to missing terms.")
                expansion_rels = [] # Found specific target, discard generic expansion
            elif len(unique_entities) == 1 and not bridge_neighbor_ids:
                logs.append("DEBUG: Missing terms not found in neighborhood. Pruning generic expansion.")
                expansion_rels = [] # Avoid noise if we can't find the second entity

        final_rels = list(direct_rels) + targeted_rels
        
        # Dynamic Expansion Limit: If we have strong signals (Direct or Bridge), reduce generic noise.
        has_strong_signal = len(direct_rels) > 0 or len(bridge_neighbor_ids) > 0 or len(targeted_rels) > 0
        MAX_EXPANSION = 2 if has_strong_signal else 5
        
        rels_by_entity = defaultdict(list)
        for r in expansion_rels:
            src_id = r.source_entity_id
            tgt_id = r.target_entity_id
            neighbor_id = tgt_id if src_id in unique_entities else src_id
            
            # Always include bridge relationships, otherwise subject to limit
            if neighbor_id in bridge_neighbor_ids:
                final_rels.append(r)
            else:
                anchor_id = src_id if src_id in unique_entities else tgt_id
                rels_by_entity[anchor_id].append(r)
        
        for anchor_id, rels in rels_by_entity.items():
            # Discard overly generic edges (RELATED_TO) for expansion nodes to reduce noise
            filtered_rels = [r for r in rels if r.relationship_type != 'RELATED_TO']
            
            # Deprioritize 'RELATED_TO', then sort by confidence
            filtered_rels.sort(key=lambda x: (x.relationship_type == 'RELATED_TO', -x.confidence_score))
            final_rels.extend(filtered_rels[:MAX_EXPANSION])
            
        # 4. Format Output
        
        # Canonicalization: Map multiple IDs to a single representative ID for (Name, Type)
        canonical_map = {} # (name, type) -> canonical_id
        id_to_canonical = {} # raw_id -> canonical_id
        
        # Build map from all unique entities found
        for eid, e in unique_entities.items():
            key = (e.entity_text, e.entity_type)
            if key not in canonical_map:
                canonical_map[key] = eid
            id_to_canonical[eid] = canonical_map[key]

        formatted_rels = []
        final_canonical_ids = set()
        seen_edges = set()
        
        for r in final_rels:
            src_canon = id_to_canonical.get(r.source_entity_id)
            tgt_canon = id_to_canonical.get(r.target_entity_id)
            
            if src_canon and tgt_canon and src_canon != tgt_canon:
                edge_key = (src_canon, tgt_canon, r.relationship_type)
                if edge_key not in seen_edges:
                    final_canonical_ids.add(src_canon)
                    final_canonical_ids.add(tgt_canon)
                    formatted_rels.append({
                        "source_id": str(src_canon),
                        "target_id": str(tgt_canon),
                        "type": r.relationship_type
                    })
                    seen_edges.add(edge_key)
            
        # Resolve names for all involved entities (including new neighbors)
        missing_ids = final_entity_ids - set(unique_entities.keys())
        if missing_ids:
            stmt_missing = select(Entity).filter(Entity.entity_id.in_(missing_ids))
            neighbor_entities = self.session.execute(stmt_missing).scalars().all()
            for e in neighbor_entities:
                unique_entities[e.entity_id] = e
        
        formatted_entities = []
        seen_entities = set()
        for eid, e in unique_entities.items():
            # Only include if this ID is the canonical one and it was used in the graph
            if eid in final_canonical_ids and eid == id_to_canonical[eid]:
                formatted_entities.append({
                    "id": str(eid), # Provide the canonical ID
                    "name": e.entity_text, 
                    "type": e.entity_type
                })
        
        logs.append(f"DEBUG: KG Search returned {len(formatted_rels)} relationships (Direct: {len(direct_rels)}).")
        return {"entities": formatted_entities, "relationships": formatted_rels}, logs

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