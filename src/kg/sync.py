import os
import logging
import time
from typing import List, Dict, Any
from sqlalchemy import select
from neo4j import GraphDatabase
from dotenv import load_dotenv

from src.db.session import get_session
from src.db.models import Entity, Relationship

# Load env vars if running standalone
load_dotenv(dotenv_path="cfg/.env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jSyncAgent:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def sync(self):
        """Main execution method."""
        start_time = time.time()
        self.connect()
        pg_session = get_session()
        
        try:
            # 1. Fetch Data from Postgres
            logger.info("Fetching data from PostgreSQL...")
            entities = pg_session.execute(select(Entity)).scalars().all()
            relationships = pg_session.execute(select(Relationship)).scalars().all()
            
            logger.info(f"Postgres State: {len(entities)} Entities, {len(relationships)} Relationships.")

            # 2. Sync Nodes
            self._sync_nodes(entities)
            
            # 3. Sync Relationships
            self._sync_relationships(relationships)
            
            # 4. Cleanup (Deletions)
            self._prune_orphans(entities, relationships)
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"Sync completed in {duration}ms.")
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise
        finally:
            pg_session.close()
            self.close()

    def _sync_nodes(self, entities: List[Entity]):
        """Upsert entities as Nodes."""
        if not entities:
            return

        query = """
        UNWIND $batch AS row
        MERGE (n:Entity {id: row.id})
        SET n.name = row.name,
            n.type = row.type,
            n.confidence = row.confidence,
            n.source = 'postgres'
        """
        
        batch_data = [
            {
                "id": str(e.entity_id),
                "name": e.entity_text,
                "type": e.entity_type,
                "confidence": e.confidence_score
            }
            for e in entities
        ]
        
        with self.driver.session() as session:
            session.run(query, batch=batch_data)
            logger.info(f"Upserted {len(batch_data)} nodes in Neo4j.")

    def _sync_relationships(self, relationships: List[Relationship]):
        """Upsert relationships as Edges."""
        if not relationships:
            return

        # Group by type to handle dynamic relationship types in Cypher
        grouped = {}
        for r in relationships:
            r_type = self._sanitize_rel_type(r.relationship_type)
            if r_type not in grouped:
                grouped[r_type] = []
            grouped[r_type].append({
                "id": str(r.relationship_id),
                "source": str(r.source_entity_id),
                "target": str(r.target_entity_id),
                "confidence": r.confidence_score
            })

        with self.driver.session() as session:
            count = 0
            for r_type, batch in grouped.items():
                # Dynamic Cypher construction for Relationship Type
                query = f"""
                UNWIND $batch AS row
                MATCH (s:Entity {{id: row.source}})
                MATCH (t:Entity {{id: row.target}})
                MERGE (s)-[r:{r_type} {{id: row.id}}]->(t)
                SET r.confidence = row.confidence
                """
                session.run(query, batch=batch)
                count += len(batch)
            logger.info(f"Upserted {count} relationships in Neo4j.")

    def _prune_orphans(self, entities: List[Entity], relationships: List[Relationship]):
        """Remove Nodes/Edges in Neo4j that no longer exist in Postgres."""
        valid_node_ids = [str(e.entity_id) for e in entities]
        valid_rel_ids = [str(r.relationship_id) for r in relationships]
        
        with self.driver.session() as session:
            # Prune Edges
            if valid_rel_ids:
                session.run("""
                MATCH ()-[r]->()
                WHERE NOT r.id IN $valid_ids
                DELETE r
                """, valid_ids=valid_rel_ids)
            else:
                # If no relationships in PG, delete all in Neo4j
                session.run("MATCH ()-[r]->() DELETE r")
                
            # Prune Nodes
            if valid_node_ids:
                session.run("""
                MATCH (n:Entity)
                WHERE NOT n.id IN $valid_ids
                DETACH DELETE n
                """, valid_ids=valid_node_ids)
            else:
                session.run("MATCH (n:Entity) DETACH DELETE n")
                
            logger.info("Pruning complete.")

    def _sanitize_rel_type(self, text: str) -> str:
        """Ensure relationship type is safe for Cypher."""
        # Replace spaces with underscores, uppercase, remove non-alphanumeric
        safe = "".join(c if c.isalnum() else "_" for c in text)
        return safe.upper()

if __name__ == "__main__":
    agent = Neo4jSyncAgent()
    agent.sync()