import os
import logging
import time
import uuid
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
        
        # Generate a unique ID for this sync run to track active nodes
        sync_run_id = str(uuid.uuid4())
        
        try:
            # 1. Fetch Data from Postgres
            logger.info("Fetching data from PostgreSQL...")
            entities = pg_session.execute(select(Entity)).scalars().all()
            relationships = pg_session.execute(select(Relationship)).scalars().all()
            
            logger.info(f"Postgres State: {len(entities)} Entities, {len(relationships)} Relationships.")

            # 1.5 Create Indexes
            self._create_indexes()

            # 2. Sync Nodes
            self._sync_nodes(entities, sync_run_id)
            
            # 3. Sync Relationships
            self._sync_relationships(relationships, entities, sync_run_id)
            
            # 4. Cleanup (Deletions)
            self._prune_orphans(sync_run_id)
            
            duration = int((time.time() - start_time) * 1000)
            logger.info(f"Sync completed in {duration}ms.")
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise
        finally:
            pg_session.close()
            self.close()

    def _create_indexes(self):
        """Create performance indexes in Neo4j."""
        queries = [
            # We remove the ID constraint because multiple Postgres IDs now map to one Neo4j node
            "DROP CONSTRAINT FOR (n:Entity) REQUIRE n.id IS UNIQUE IF EXISTS",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)"
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.warning(f"Index operation failed (might be expected): {e}")
        logger.info("Neo4j indexes verified.")

    def _sync_nodes(self, entities: List[Entity], sync_id: str):
        """Upsert entities as Nodes."""
        if not entities:
            return

        query = """
        UNWIND $batch AS row
        MERGE (n:Entity {name: row.name, type: row.type})
        SET n.confidence = row.confidence,
            n.source = 'postgres',
            n.last_sync = $sync_id
        """
        
        batch_data = [
            {
                "name": e.entity_text,
                "type": e.entity_type,
                "confidence": e.confidence_score
            }
            for e in entities
        ]
        
        with self.driver.session() as session:
            session.run(query, batch=batch_data, sync_id=sync_id)
            logger.info(f"Upserted {len(batch_data)} nodes in Neo4j.")

    def _sync_relationships(self, relationships: List[Relationship], entities: List[Entity], sync_id: str):
        """Upsert relationships as Edges."""
        if not relationships:
            return
            
        # Build lookup for Entity ID -> (Name, Type)
        entity_lookup = {e.entity_id: (e.entity_text, e.entity_type) for e in entities}

        # Group by type to handle dynamic relationship types in Cypher
        grouped = {}
        for r in relationships:
            src_info = entity_lookup.get(r.source_entity_id)
            tgt_info = entity_lookup.get(r.target_entity_id)
            
            if not src_info or not tgt_info:
                continue
                
            r_type = self._sanitize_rel_type(r.relationship_type)
            if r_type not in grouped:
                grouped[r_type] = []
            grouped[r_type].append({
                "source_name": src_info[0], "source_type": src_info[1],
                "target_name": tgt_info[0], "target_type": tgt_info[1],
                "confidence": r.confidence_score
            })

        with self.driver.session() as session:
            count = 0
            for r_type, batch in grouped.items():
                # Dynamic Cypher construction for Relationship Type
                query = f"""
                UNWIND $batch AS row
                MATCH (s:Entity {{name: row.source_name, type: row.source_type}})
                MATCH (t:Entity {{name: row.target_name, type: row.target_type}})
                MERGE (s)-[r:{r_type}]->(t)
                SET r.confidence = row.confidence,
                    r.last_sync = $sync_id
                """
                session.run(query, batch=batch, sync_id=sync_id)
                count += len(batch)
            logger.info(f"Upserted {count} relationships in Neo4j.")

    def _prune_orphans(self, sync_id: str):
        """Remove Nodes/Edges in Neo4j that were not updated in this sync run."""
        with self.driver.session() as session:
            # Prune Edges
            session.run("""
            MATCH ()-[r]->()
            WHERE r.last_sync <> $sync_id
            DELETE r
            """, sync_id=sync_id)
                
            # Prune Nodes
            session.run("""
            MATCH (n:Entity)
            WHERE n.last_sync <> $sync_id
            DETACH DELETE n
            """, sync_id=sync_id)
                
            logger.info("Pruning complete.")

    def _sanitize_rel_type(self, text: str) -> str:
        """Ensure relationship type is safe for Cypher."""
        # Replace spaces with underscores, uppercase, remove non-alphanumeric
        safe = "".join(c if c.isalnum() else "_" for c in text)
        return safe.upper()

if __name__ == "__main__":
    agent = Neo4jSyncAgent()
    agent.sync()