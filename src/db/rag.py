import logging
import time
from typing import Dict, Any

from src.db.engine import RetrievalEngine
from src.db.ollama_client import OllamaClient

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# RAG Pipeline Orchestrator
class RAGPipeline:
    def __init__(self):
        self.retriever = RetrievalEngine()
        self.llm = OllamaClient() # Defaults to mistral, configurable via env
        
        # Check model availability once at startup
        if not self.llm.check_model_availability():
            logger.warning("Ollama model check failed at startup. Pipeline may not function correctly.")
        
    def run(self, query: str) -> Dict[str, Any]:
        """
        End-to-end RAG execution: Retrieve -> Format -> Generate.
        """
        start_time = time.time()
        logger.info(f"Starting RAG pipeline for query: '{query}'")
        
        # 1. Retrieval Phase
        retrieval_results = self.retriever.retrieve(query)
        
        # 3. Context Formatting
        chunks = retrieval_results.get("chunks", [])
        graph = retrieval_results.get("graph", {})
        context_str = self.llm.format_context(chunks, graph)
        
        # 4. Generation Phase
        answer = self.llm.generate_answer(query, context_str)
        
        duration = int((time.time() - start_time) * 1000)
        logger.info(f"RAG pipeline completed in {duration}ms")
        
        return {
            "query": query,
            "answer": answer,
            "context_used": context_str,
            "raw_retrieval": retrieval_results,
            "execution_time_ms": duration
        }