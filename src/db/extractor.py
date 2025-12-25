import json
import uuid
import logging
import os
from typing import Dict, Any
from sqlalchemy.orm import Session
import spacy
from sqlalchemy import select

# Import database utilities
from .session import get_session

from .models import Chunk, Entity, Relationship

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def spacy_extract(text: str) -> Dict[str, Any]:
    """
    Extracts entities and relationships using spaCy (Offline/Local).
    """
    doc = nlp(text)
    
    entities = []
    relationships = []
    
    # 1. Extract Entities
    # Map token indices to entities to help with relationship building
    entity_map = {} 
    
    # Filter for specific entity types relevant to a KG
    valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC", "FAC"}
    
    for ent in doc.ents:
        if ent.label_ in valid_labels:
            ent_data = {"name": ent.text, "type": ent.label_}
            entities.append(ent_data)
            # Map every token in the entity span to the entity data
            for token in ent:
                entity_map[token.i] = ent_data

    # 2. Extract Relationships (Rule-based Dependency Parsing)
    # Heuristic: Subject -> Verb -> Object
    for token in doc:
        # Look for main verbs
        if token.pos_ == "VERB":
            subj = None
            obj = None
            
            # Find Subject (nsubj: nominal subject, nsubjpass: passive nominal subject)
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child
                    break
            
            # Find Object (dobj: direct object, pobj: object of preposition, attr: attribute)
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child
                    break
            
            # Check if subject and object resolve to extracted entities
            if subj and obj:
                subj_ent = entity_map.get(subj.i)
                obj_ent = entity_map.get(obj.i)
                
                # Only create relationship if both ends are named entities and distinct
                if subj_ent and obj_ent and subj_ent != obj_ent:
                    relationships.append({
                        "source": subj_ent["name"],
                        "target": obj_ent["name"],
                        "type": token.lemma_ # Use the lemma of the verb (e.g., "acquired")
                    })

    return {
        "entities": entities,
        "relationships": relationships
    }

def extract_and_store_graph(chunk_id: uuid.UUID):
    """
    Main pipeline function:
    1. Gets Chunk -> 2. Runs spaCy -> 3. Saves Entities -> 4. Saves Relationships
    """
    session = get_session()
    try:
        # 1. Fetch the chunk
        stmt = select(Chunk).where(Chunk.chunk_id == chunk_id)
        chunk = session.execute(stmt).scalars().first()
        
        if not chunk:
            logger.error(f"Chunk {chunk_id} not found.")
            return

        logger.info(f"Processing Chunk {chunk_id}...")

        # 2. Run spaCy Extraction
        graph_data = spacy_extract(chunk.chunk_text or "")
        
        # 3. Process Entities
        entity_name_to_id = {}
        
        entities_data = graph_data.get("entities", [])
        for ent in entities_data:
            name = ent.get("name")
            etype = ent.get("type", "Unknown")
            
            if not name:
                continue
                
            new_entity = Entity(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                block_id=chunk.block_id,
                page_number=chunk.page_number,
                entity_text=name,
                entity_type=etype,
                confidence_score=90,
                extraction_method="spacy_ner",
                metadata_json={"source": "spacy_pipeline"}
            )
            session.add(new_entity)
            session.flush() # Flush to generate the new_entity.entity_id immediately
            
            entity_name_to_id[name] = new_entity.entity_id
            
        # 4. Process Relationships
        rels_data = graph_data.get("relationships", [])
        for rel in rels_data:
            src = rel.get("source")
            tgt = rel.get("target")
            rtype = rel.get("type", "related_to")
            
            src_id = entity_name_to_id.get(src)
            tgt_id = entity_name_to_id.get(tgt)
            
            if src_id and tgt_id:
                new_rel = Relationship(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    block_id=chunk.block_id,
                    page_number=chunk.page_number,
                    source_entity_id=src_id,
                    target_entity_id=tgt_id,
                    relationship_type=rtype,
                    confidence_score=80,
                    extraction_method="spacy_dep_parse"
                )
                session.add(new_rel)
            else:
                # Common in rule-based systems, logging every skip might be noisy
                pass

        session.commit()
        logger.info(f"Graph extraction complete. Saved {len(entities_data)} entities and {len(rels_data)} relationships.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error processing graph: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    # Simple test runner
    session = get_session()
    # Fetch all chunk IDs
    stmt = select(Chunk.chunk_id)
    all_chunk_ids = session.execute(stmt).scalars().all()
    session.close()
    
    if all_chunk_ids:
        print(f"Found {len(all_chunk_ids)} chunks. Starting extraction...")
        for i, chunk_id in enumerate(all_chunk_ids):
            print(f"[{i+1}/{len(all_chunk_ids)}] Processing chunk {chunk_id}...")
            extract_and_store_graph(chunk_id)
    else:
        print("No chunks found in the database. Please ingest a document first.")