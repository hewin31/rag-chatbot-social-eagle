"""Neo4j connection and utilities for knowledge graph storage and queries."""
import os
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from src.config import logger


class Neo4jDriver:
    """Manages Neo4j connection and graph operations."""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize Neo4j driver.
        Defaults to environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def create_entity_node(self, entity_id: str, entity_text: str, entity_type: str, 
                          document_id: str, chunk_id: str, page_number: int,
                          confidence: float, metadata: Dict = None) -> bool:
        """Create an entity node in Neo4j."""
        if not self.driver:
            return False
        
        query = """
        MERGE (e:Entity {entity_id: $entity_id})
        SET e.text = $text, e.type = $type, e.confidence = $confidence,
            e.document_id = $doc_id, e.chunk_id = $chunk_id, e.page = $page,
            e.metadata = $metadata
        RETURN e
        """
        try:
            with self.driver.session() as session:
                session.run(query, {
                    "entity_id": entity_id,
                    "text": entity_text,
                    "type": entity_type,
                    "doc_id": document_id,
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "confidence": confidence,
                    "metadata": metadata or {}
                })
            return True
        except Exception as e:
            logger.error(f"Error creating entity node: {e}")
            return False
    
    def create_relationship(self, rel_id: str, source_entity_id: str, target_entity_id: str,
                           rel_type: str, rel_text: str, confidence: float,
                           document_id: str, chunk_id: str, page_number: int,
                           metadata: Dict = None) -> bool:
        """Create a relationship between two entities."""
        if not self.driver:
            return False
        
        query = """
        MATCH (source:Entity {entity_id: $source_id})
        MATCH (target:Entity {entity_id: $target_id})
        MERGE (source)-[r:REL {rel_id: $rel_id}]->(target)
        SET r.type = $rel_type, r.text = $rel_text, r.confidence = $confidence,
            r.document_id = $doc_id, r.chunk_id = $chunk_id, r.page = $page,
            r.metadata = $metadata
        RETURN r
        """
        try:
            with self.driver.session() as session:
                session.run(query, {
                    "rel_id": rel_id,
                    "source_id": source_entity_id,
                    "target_id": target_entity_id,
                    "rel_type": rel_type,
                    "rel_text": rel_text,
                    "confidence": confidence,
                    "doc_id": document_id,
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "metadata": metadata or {}
                })
            return True
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    def query_entities_by_type(self, entity_type: str, limit: int = 10) -> List[Dict]:
        """Query entities by type."""
        if not self.driver:
            return []
        
        query = "MATCH (e:Entity {type: $type}) RETURN e LIMIT $limit"
        try:
            with self.driver.session() as session:
                result = session.run(query, {"type": entity_type, "limit": limit})
                return [dict(record["e"]) for record in result]
        except Exception as e:
            logger.error(f"Error querying entities: {e}")
            return []
    
    def query_neighbors(self, entity_id: str, depth: int = 1) -> Dict:
        """Query entities connected to a given entity (graph traversal)."""
        if not self.driver:
            return {}
        
        query = f"""
        MATCH (e:Entity {{entity_id: $entity_id}})
        MATCH (e)-[r*1..{depth}]-(neighbor)
        RETURN e, r, neighbor
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, {"entity_id": entity_id})
                neighbors = []
                for record in result:
                    neighbors.append({
                        "source": dict(record["e"]),
                        "relationships": [dict(rel) for rel in record["r"]] if record["r"] else [],
                        "target": dict(record["neighbor"])
                    })
                return {"entity": entity_id, "neighbors": neighbors}
        except Exception as e:
            logger.error(f"Error querying neighbors: {e}")
            return {}
    
    def clear_document_graph(self, document_id: str) -> bool:
        """Remove all entities and relationships for a document (for re-extraction)."""
        if not self.driver:
            return False
        
        query = """
        MATCH (e:Entity {document_id: $doc_id})
        DETACH DELETE e
        """
        try:
            with self.driver.session() as session:
                session.run(query, {"doc_id": document_id})
            logger.info(f"Cleared graph for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing document graph: {e}")
            return False