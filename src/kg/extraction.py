"""Entity and relationship extraction from chunks using LLM."""
import json
import uuid
from typing import Dict
import openai
from src.config import logger
from src.db import get_session
from src.db.models import Chunk, Entity, Relationship


def extract_entities_and_relations(chunk_text: str, chunk_id: str, block_id: str, 
                                   document_id: str, page_number: int,
                                   model: str = "gpt-3.5-turbo") -> Dict:
    """
    Use LLM to extract entities and relationships from a chunk.
    
    Returns: {
        "entities": [{"text": str, "type": str}, ...],
        "relationships": [{"source": str, "target": str, "type": str, "text": str}, ...]
    }
    """
    prompt = f"""
You are a knowledge graph extraction assistant. Extract named entities and their relationships from the following text.

Output ONLY a JSON object with this structure (no markdown, no explanation):
{{
  "entities": [
    {{"text": "entity_text", "type": "PERSON|ORG|LOCATION|REGULATION|PRODUCT|OTHER"}},
    ...
  ],
  "relationships": [
    {{"source": "entity_text_1", "target": "entity_text_2", "type": "REL_TYPE", "text": "relationship_description"}},
    ...
  ]
}}

Relationship types: CEO_OF, FOUNDED, LOCATED_IN, REGULATES, REFERENCES, OWNS, WORKS_FOR, REQUIRES, IMPLEMENTS, VIOLATES, ASSOCIATED_WITH

TEXT:
{chunk_text[:2000]}
"""
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        # Try to parse JSON; if embedded in markdown, extract it
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM extraction JSON: {e}")
        return {"entities": [], "relationships": []}
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"entities": [], "relationships": []}


def extract_kg_for_document(document_id: str, confidence_threshold: int = 85,
                            model: str = "gpt-3.5-turbo") -> Dict:
    """
    Extract all entities and relationships from a document's chunks.
    Store in SQL (entities, relationships tables).
    
    Returns summary: {document_id, entities_created, relationships_created}
    """
    with get_session() as session:
        # Fetch high-confidence chunks
        chunks = session.query(Chunk).filter(
            Chunk.document_id == document_id,
            Chunk.confidence_score >= confidence_threshold
        ).all()
        
        entities_created = 0
        relationships_created = 0
        entity_map = {}  # text -> entity_id for linking relationships
        
        # Extract entities and relationships from each chunk
        for chunk in chunks:
            result = extract_entities_and_relations(
                chunk.chunk_text,
                str(chunk.chunk_id),
                str(chunk.block_id),
                str(document_id),
                chunk.page_number,
                model=model
            )
            
            # Create entity records
            for ent in result.get("entities", []):
                entity_id = str(uuid.uuid4())
                entity = Entity(
                    entity_id=entity_id,
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    block_id=chunk.block_id,
                    page_number=chunk.page_number,
                    entity_text=ent["text"],
                    entity_type=ent.get("type", "OTHER"),
                    confidence_score=85,
                    extraction_method="llm_ner",
                    metadata_json={"chunk_text_snippet": chunk.chunk_text[:200]}
                )
                session.add(entity)
                entity_map[ent["text"]] = entity_id
                entities_created += 1
            
            # Create relationship records
            for rel in result.get("relationships", []):
                source_text = rel["source"]
                target_text = rel["target"]
                
                # Skip if entities weren't extracted in this batch
                if source_text not in entity_map or target_text not in entity_map:
                    continue
                
                relationship = Relationship(
                    relationship_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    block_id=chunk.block_id,
                    page_number=chunk.page_number,
                    source_entity_id=entity_map[source_text],
                    target_entity_id=entity_map[target_text],
                    relationship_type=rel.get("type", "ASSOCIATED_WITH"),
                    relationship_text=rel.get("text", ""),
                    confidence_score=80,
                    extraction_method="llm_relation_extraction",
                    metadata_json={"chunk_id": str(chunk.chunk_id)}
                )
                session.add(relationship)
                relationships_created += 1
        
        session.commit()
    
    return {
        "document_id": str(document_id),
        "entities_created": entities_created,
        "relationships_created": relationships_created
    }


def sync_kg_to_neo4j(document_id: str, neo4j_driver) -> Dict:
    """Sync SQL entities and relationships to Neo4j."""
    with get_session() as session:
        entities = session.query(Entity).filter(Entity.document_id == document_id).all()
        relationships = session.query(Relationship).filter(Relationship.document_id == document_id).all()
        
        entities_synced = 0
        for ent in entities:
            success = neo4j_driver.create_entity_node(
                str(ent.entity_id),
                ent.entity_text,
                ent.entity_type,
                str(ent.document_id),
                str(ent.chunk_id),
                ent.page_number,
                ent.confidence_score / 100.0,
                metadata=ent.metadata_json
            )
            if success:
                entities_synced += 1
        
        rels_synced = 0
        for rel in relationships:
            success = neo4j_driver.create_relationship(
                str(rel.relationship_id),
                str(rel.source_entity_id),
                str(rel.target_entity_id),
                rel.relationship_type,
                rel.relationship_text,
                rel.confidence_score / 100.0,
                str(rel.document_id),
                str(rel.chunk_id),
                rel.page_number,
                metadata=rel.metadata_json
            )
            if success:
                rels_synced += 1
    
    return {
        "document_id": str(document_id),
        "entities_synced": entities_synced,
        "relationships_synced": rels_synced
    }