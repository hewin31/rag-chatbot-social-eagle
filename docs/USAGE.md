# Usage

## Setup

Make sure to configure `cfg/.env.example` as `cfg/.env` and set `POSTGRES_DSN`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create database tables (SQLAlchemy):

```bash
python -c "from src.db import engine; from src.db.models import Base; Base.metadata.create_all(bind=engine); print('tables created')"
```

---

## Phase 1: Ingest PDF

Ingest a PDF file:

```bash
python -m src.ingest.ingest_cli ingest /path/to/sample.pdf
# Output: Ingested document_id=<uuid> path=data/pdfs/<uuid>/sample.pdf
```

---

## Phase 2: Probe

Probe a document by its document_id:

```bash
python -m src.probe.cli probe <document_id> --samples 5
# Output: JSON summary with complexity_score, recommended_action, per_page signals
```

---

## Phase 3: Parse

Parse a document (extract text and tables):

```bash
python -m src.ingest.parse_cli parse <document_id>
# Output: Parsed document <document_id>; inserted <N> blocks; status=parsed
```

---

## Phase 4: Verify & Audit

Audit a document (SQL-based checks):

```bash
python -m src.verify.cli audit <document_id>
# Output: JSON report with block counts, confidence stats, extraction methods, issues
```

Cross-validate blocks against source PDF:

```bash
python -m src.verify.cli validate <document_id> --samples 3
# Output: JSON validation report; checks ±10% variance
```

List all documents and their ingestion status:

```bash
python -m src.verify.cli status
# Output: JSON array with filename, status, page_count, blocks_extracted
```

---

## Phase 5: Adaptive Chunking

Adaptively chunk blocks into semantic units:

```bash
python -m src.ingest.chunk_cli chunk <document_id>
# Output: Chunked document <document_id>; created <N> chunks
```

Verify chunks:

```bash
python -m src.verify.chunk_checks verify <document_id>
# Output: JSON report with chunk distribution, token stats, issues
```

---

## End-to-End Example

```bash
# 1. Ingest
python -m src.ingest.ingest_cli ingest ./sample.pdf
DOC_ID=<printed_id>

# 2. Probe
python -m src.probe.cli probe $DOC_ID

# 3. Parse
python -m src.ingest.parse_cli parse $DOC_ID

# 4. Verify
python -m src.verify.cli audit $DOC_ID
python -m src.verify.cli validate $DOC_ID

# 5. Chunk
python -m src.ingest.chunk_cli chunk $DOC_ID

# 6. Verify chunks
python -m src.verify.chunk_checks verify $DOC_ID
```
