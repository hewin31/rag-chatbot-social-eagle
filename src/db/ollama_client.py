import logging
import os
import time
from typing import List, Dict, Any
import ollama

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class OllamaClient:
    def __init__(self, model_name: str = "phi3"):
        # Allow override via env var, default to phi3
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        
    def check_model_availability(self) -> bool:
        """Checks if Ollama is running and the model is available."""
        try:
            # Check connectivity and list models
            response = ollama.list()
            
            # Handle response being an object (newer library) or dict (older/raw)
            if hasattr(response, 'models'):
                models = response.models
            else:
                models = response.get('models', [])
            
            available_models = []
            for m in models:
                # Handle model item being an object or dict
                if hasattr(m, 'name'):
                    available_models.append(m.name)
                elif isinstance(m, dict):
                    available_models.append(m.get('name') or m.get('model'))
                else:
                    available_models.append(str(m))
            
            # Check if our model (or a version of it) exists
            # We look for partial match (e.g. "mistral" in "mistral:latest")
            model_exists = any(self.model_name in m for m in available_models)
            
            if not model_exists:
                logger.info(f"Model '{self.model_name}' not found locally. Pulling (this may take time)...")
                ollama.pull(self.model_name)
                logger.info(f"Model '{self.model_name}' pulled successfully.")
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}. Ensure 'ollama serve' is running.")
            return False

    def format_context(self, chunks: List[Dict], graph_data: Dict[str, Any]) -> str:
        """Formats retrieved data into a string for the LLM prompt."""
        context_parts = []
        
        # 1. Format Text Chunks
        # Limit to top 3 chunks to prevent context overflow
        top_chunks = chunks[:3] if chunks else []
        
        if top_chunks:
            context_parts.append("--- RELEVANT TEXT EXCERPTS ---")
            for chunk in top_chunks:
                # Handle both vector search and sql fallback keys
                source_id = chunk.get('chunk_id', 'Unknown')
                text = chunk.get('text', '').strip()
                
                # Hard limit character count per chunk
                if len(text) > 500:
                    text = text[:500] + "...(truncated)"
                    
                context_parts.append(f"[Source: {source_id}]\n{text}\n")
        
        # 2. Format Knowledge Graph
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        if relationships and entities:
            context_parts.append("--- KNOWLEDGE GRAPH CONNECTIONS ---")
            
            # Create ID lookup map
            id_to_name = {e['id']: e['name'] for e in entities}
            
            # Limit relationships to keep prompt concise
            for i, rel in enumerate(relationships):
                if i >= 15: # Max 15 relationships
                    break
                src_id = rel.get('source_id')
                tgt_id = rel.get('target_id')
                r_type = rel.get('type')
                
                src_name = id_to_name.get(src_id, src_id)
                tgt_name = id_to_name.get(tgt_id, tgt_id)
                
                context_parts.append(f"{src_name} --[{r_type}]--> {tgt_name}")
                
        if not context_parts:
            return "No relevant context found."
            
        return "\n\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> str:
        """Generates an answer using the local LLM."""
        system_prompt = (
            "You are a helpful financial assistant. "
            "Use ONLY the provided Context to answer the user's Question. "
            "If the answer is not found in the Context, explicitly state that you do not know. "
            "Do not hallucinate or use outside knowledge. "
            "Cite the source IDs (e.g., [Source: ...]) if applicable."
        )
        
        user_message = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        logger.info(f"Sending prompt to LLM (Context size: {len(context)} chars)...")
        start_gen = time.time()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message},
                ]
            )
            duration = time.time() - start_gen
            logger.info(f"LLM response received in {duration:.2f}s")
            return response['message']['content']
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return "I encountered an error while generating the answer."
