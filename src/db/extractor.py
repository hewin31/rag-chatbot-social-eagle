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

# --- CONFIGURATION: Domain Dictionary ---
ENTITY_CONFIG = {
    "ORG": ["company", "organization", "bank", "institution", "fund", "branch", "location", "goldman sachs", "icici bank"],
    "SECURITY": ["stock", "security", "share", "bond", "etf", "treasury", "equity", "instrument", "option", "future", "derivative", "asset", "aapl", "nifty"],
    "CURRENCY": ["currency", "dollar", "rupee", "euro", "usd", "inr", "eur"],
    "ACCOUNT": ["account", "savings", "brokerage"],
    "TRANSACTION": ["transaction", "wire", "transfer", "payment", "deposit", "withdrawal"],
    "FINANCIAL_METRIC": ["revenue", "expense", "cost", "profit", "loss", "income"],
    "TAX": ["tax", "fee", "charge", "gst", "deduction", "exemption", "refund", "assessment", "return", "filing"],
    "DEBT": ["loan", "debt", "borrowing", "mortgage", "interest", "credit"],
    "PAYMENT_METHOD": ["card", "neft", "cheque"],
    "CONTRACT": ["contract", "agreement", "policy", "insurance"],
    "RATING": ["rating", "score"],
    "RISK": ["risk", "market"],
    "COMMODITY": ["commodity", "gold", "oil", "wheat"],
    "INDEX": ["index", "sensex", "nifty"],
    "REGULATION": ["compliance", "regulation", "audit", "act", "section", "penalty", "penalties"]
}

RELATIONSHIP_CONFIG = {
    "own": "OWNS", "hold": "OWNS", "manage": "MANAGES", "operate": "MANAGES", "belong": "PART_OF", "part": "PART_OF",
    "invest": "INVESTS_IN", "invest_in": "INVESTS_IN", "fund": "FUNDED_BY", "fund_by": "FUNDED_BY",
    "owe": "OWES", "debtor": "DEBTOR_OF", "secure": "SECURED_BY", "secure_by": "SECURED_BY",
    "transact": "TRANSACTED_WITH", "transact_with": "TRANSACTED_WITH", "pay": "PAYS", "receive": "RECEIVES",
    "rate": "RATED_BY", "rate_by": "RATED_BY", "evaluate": "EVALUATED_BY", "accrue": "INCURRED", "incur": "INCURRED",
    "generate": "GENERATES", "yield": "GENERATES", "convert": "CONVERTED_TO", "convert_to": "CONVERTED_TO",
    "trade": "TRADED_ON", "trade_on": "TRADED_ON", "list": "LISTED_AT",
    "subject": "SUBJECT_TO", "subject_to": "SUBJECT_TO", "comply": "COMPLIANT_WITH", "comply_with": "COMPLIANT_WITH",
    "cover": "COVERED_BY", "cover_by": "COVERED_BY", "audit": "AUDITED_BY", "verify": "VERIFIED_BY"
}

# Add EntityRuler to enforce these terms
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    for label, terms in ENTITY_CONFIG.items():
        for term in terms:
            patterns.append({"label": label, "pattern": [{"LOWER": term.lower()}]})
    ruler.add_patterns(patterns)

def _normalize_entity_text(text: str, label: str) -> str:
    """
    Normalizes entity text to reduce duplicates (e.g., 'Tax' -> 'tax').
    """
    text = text.strip()
    
    # 1. Always lowercase domain concepts and common financial terms
    concept_labels = {
        "SECURITY", "CURRENCY", "ACCOUNT", "TRANSACTION", "FINANCIAL_METRIC", 
        "TAX", "DEBT", "PAYMENT_METHOD", "CONTRACT", "RATING", "RISK", 
        "COMMODITY", "INDEX", "REGULATION", "DOMAIN_CONCEPT"
    }
    
    if label in concept_labels:
        return text.lower()
        
    # 2. For ORG, lowercase only if it's a generic term
    if label == "ORG" and text.lower() in ["company", "organization", "bank", "institution", "fund", "branch", "location"]:
        return text.lower()
        
    # 3. Default: Return as-is (preserves 'India', 'Deloitte', etc.)
    return text

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
    valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LOC", "FAC"}.union(ENTITY_CONFIG.keys())
    
    for ent in doc.ents:
        if ent.label_ in valid_labels:
            norm_name = _normalize_entity_text(ent.text, ent.label_)
            ent_data = {"name": norm_name, "type": ent.label_}
            entities.append(ent_data)
            # Map every token in the entity span to the entity data
            for token in ent:
                entity_map[token.i] = ent_data

    # Helper to resolve conjunctions (e.g., "Alice and Bob")
    def resolve_entities(token):
        found = []
        if token.i in entity_map:
            found.append(entity_map[token.i])
        for child in token.children:
            if child.dep_ == "conj":
                found.extend(resolve_entities(child))
        return found

    # 2. Extract Relationships (Enhanced Dependency Parsing)
    for token in doc:
        
        # Case A: Verbs (Active & Passive)
        if token.pos_ == "VERB":
            subjects = []
            objects = []
            rel_type = None
            
            # Check for passive voice
            is_passive = False
            for child in token.children:
                if child.dep_ == "auxpass":
                    is_passive = True
                    break
            
            if is_passive:
                # Passive: Target (nsubjpass) ... by Source (agent)
                for child in token.children:
                    if child.dep_ == "nsubjpass":
                        objects.extend(resolve_entities(child))
                    if child.dep_ == "agent":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                subjects.extend(resolve_entities(grandchild))
            else:
                # Active: Source (nsubj) ... Target (dobj/attr/prep)
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subjects.extend(resolve_entities(child))
                
                for child in token.children:
                    if child.dep_ in ("dobj", "attr"):
                        objects.extend(resolve_entities(child))
                        # If direct object is an entity, use it
                        found_objs = resolve_entities(child)
                        if found_objs:
                            objects.extend(found_objs)
                        else:
                            # If not, check its children (e.g. "revealed discrepancies in REVENUE")
                            for grandchild in child.children:
                                objects.extend(resolve_entities(grandchild))
                    
                    # Prepositional objects (e.g. "invests in X")
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                # Check for compound verb in config (e.g. "invest_in")
                                compound = f"{token.lemma_.lower()}_{child.text.lower()}"
                                if compound in RELATIONSHIP_CONFIG:
                                    rel_type = RELATIONSHIP_CONFIG[compound]
                                    objects.extend(resolve_entities(grandchild))
                                elif not objects: # Fallback
                                    objects.extend(resolve_entities(grandchild))

            # Determine Relationship Type
            if not rel_type:
                lemma = token.lemma_.lower()
                rel_type = RELATIONSHIP_CONFIG.get(lemma)
            
            # Fallback for unknown verbs if we have both ends
            if not rel_type and subjects and objects:
                rel_type = lemma.upper()

            if rel_type:
                for src in subjects:
                    for tgt in objects:
                        if src != tgt:
                            relationships.append({
                                "source": src["name"],
                                "target": tgt["name"],
                                "type": rel_type
                            })

        # Case B: Possessives (Alice's account)
        if token.dep_ == "poss":
            owners = resolve_entities(token)
            assets = resolve_entities(token.head)
            for owner in owners:
                for asset in assets:
                    if owner != asset:
                        relationships.append({
                            "source": owner["name"],
                            "target": asset["name"],
                            "type": "OWNS"
                        })

        # Case C: Appositions (Alice, CEO of X)
        if token.dep_ == "appos":
            ents1 = resolve_entities(token.head)
            ents2 = resolve_entities(token)
            for e1 in ents1:
                for e2 in ents2:
                    if e1 != e2:
                        relationships.append({
                            "source": e2["name"],
                            "target": e1["name"],
                            "type": "IS_A"
                        })

        # Case D: Prepositional Noun Links (Compliance with Regulations)
        if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN"):
            sources = resolve_entities(token.head)
            targets = []
            for child in token.children:
                if child.dep_ == "pobj":
                    targets.extend(resolve_entities(child))
            
            if sources and targets:
                prep_text = token.text.lower()
                # Check config for noun_prep (e.g. "compliance_with")
                noun_lemma = token.head.lemma_.lower()
                compound = f"{noun_lemma}_{prep_text}"
                
                rtype = RELATIONSHIP_CONFIG.get(compound)
                
                if not rtype:
                    # Generic mapping
                    if prep_text == "of": rtype = "PART_OF"
                    elif prep_text == "in": rtype = "LOCATED_IN"
                    elif prep_text == "with": rtype = "ASSOCIATED_WITH"
                    elif prep_text == "for": rtype = "FOR"
                    else: rtype = "RELATED_TO"
                
                for s in sources:
                    for t in targets:
                        if s != t:
                            relationships.append({
                                "source": s["name"],
                                "target": t["name"],
                                "type": rtype
                            })
        
        # Case E: Compounds & Modifiers (e.g. "Apple stock", "High-risk loan")
        if token.dep_ in ("compound", "amod", "nmod"):
            head_ents = resolve_entities(token.head)
            child_ents = resolve_entities(token)
            for h in head_ents:
                for c in child_ents:
                    if h != c:
                        relationships.append({
                            "source": c["name"],
                            "target": h["name"],
                            "type": "MODIFIES"
                        })

    # 3. Strategy: Sentence Co-occurrence (High Recall Fallback)
    # If entities appear in the same sentence but weren't linked by grammar, link them as RELATED_TO.
    existing_pairs = set()
    for r in relationships:
        existing_pairs.add(tuple(sorted((r["source"], r["target"]))))

    for sent in doc.sents:
        sent_indices = set(range(sent.start, sent.end))
        sent_ents = []
        seen_names = set()
        
        for idx in sent_indices:
            if idx in entity_map:
                e = entity_map[idx]
                if e["name"] not in seen_names:
                    sent_ents.append(e)
                    seen_names.add(e["name"])
        
        # Link all entities in this sentence if not already linked
        if len(sent_ents) > 1:
            for i in range(len(sent_ents)):
                for j in range(i+1, len(sent_ents)):
                    e1 = sent_ents[i]
                    e2 = sent_ents[j]
                    pair = tuple(sorted((e1["name"], e2["name"])))
                    
                    if pair not in existing_pairs:
                        relationships.append({
                            "source": e1["name"],
                            "target": e2["name"],
                            "type": "RELATED_TO"
                        })
                        existing_pairs.add(pair)

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
                
            if name in entity_name_to_id:
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
        seen_rels = set()
        for rel in rels_data:
            src = rel.get("source")
            tgt = rel.get("target")
            rtype = rel.get("type", "related_to")
            
            rel_key = (src, tgt, rtype)
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)
            
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