# RAG Chatbot with Knowledge Graph

This project implements a Retrieval-Augmented Generation (RAG) pipeline using PostgreSQL (pgvector) for semantic search and Neo4j for Knowledge Graph reasoning.

## Prerequisites

1. **Services**:
   - PostgreSQL (with `pgvector` extension)
   - Neo4j Database
   - Ollama (running locally with `phi3` or similar model)

2. **Environment**:
   - Python 3.10+
   - Configure `.env` in `cfg/.env` (see `src/config.py` for variables).

## Installation

```bash
pip install -r requirements.txt
```

## Execution Steps

Follow these commands in order to ingest data, build the graph, and run the chatbot.

### 1. Initialize Database
Create the necessary tables in PostgreSQL.

```bash
python -m src.db.init_db
```

### 2. Ingest Document
Ingest a PDF file. This copies the file and creates a document record.

```bash
# Replace path with your PDF file
python -m src.ingest.ingest_cli ingest "path/to/your/document.pdf"
```
*Copy the `document_id` returned by this command (e.g., `c5c20cb9-0cfe-424e-ad81-1f288363e7ae`).*

### 3. Parse Document
Extract text and tables from the PDF.

```bash
python -m src.ingest.parse_cli parse <document_id>
```

### 4. Chunk Document
Split the parsed content into semantic chunks for vector embedding.

```bash
python -m src.ingest.chunk_cli chunk <document_id>
```

### 5. Extract Knowledge Graph
Run the NLP pipeline (spaCy) to extract entities and relationships from the text chunks.

```bash
python -m src.db.extractor
```

### 6. Sync to Neo4j
Push the extracted graph data from PostgreSQL to Neo4j for graph-based retrieval.

```bash
python -m src.kg.sync
```

### 7. Run Pipeline
Execute the RAG pipeline against the test set defined in `test.json`.

```bash
python run_pipeline.py
```

## Test Outputs

!(1.JPG)
!(C:\Users\USER\Desktop\hewin\rag chatbot\test outputs\2.JPG)
!(C:\Users\USER\Desktop\hewin\rag chatbot\test outputs\3.JPG)
!(C:\Users\USER\Desktop\hewin\rag chatbot\test outputs\4.JPG)
*(Note: Ensure your images inside the `test outputs` folder match these filenames, or update the links above.)*
!Test Output 1
!Test Output 2
!Test Output 3
!Test Output 4

*(Note: Ensure your images inside the `test outputs` folder match these filenames.)*