# RAG Document Ingestion — Phase 1 Skeleton

This repository contains the Phase‑1 skeleton for a trust-first RAG document ingestion POC.

## Design Documents

The following design documents define the overall system architecture and POC scope:

- [Rag System – File & Folder Structure Note.pdf](./Rag%20System%20%E2%80%93%20File%20&%20Folder%20Structure%20Note.pdf) — Proposed directory organization and file structure
- [Rag System – Implementation Approach Note.pdf](./Rag%20System%20%E2%80%93%20Implementation%20Approach%20Note.pdf) — Overall implementation strategy and architecture
- [Rag System – Minimal Poc Scope Note.pdf](./Rag%20System%20%E2%80%93%20Minimal%20Poc%20Scope%20Note.pdf) — Phase 1 POC requirements (document registry, probe engine, deterministic parsing, SQL lineage)
- [Rag System – Phase-level Verification Note.pdf](./Rag%20System%20%E2%80%93%20Phase-level%20Verification%20Note.pdf) — Stop-and-verify approach for each phase (5 phases total)
- [Rag Document Ingestion & Retrieval – Design Note.pdf](./Rag%20Document%20Ingestion%20&%20Retrieval%20%E2%80%93%20Design%20Note.pdf) — Comprehensive system design
- [Rag System – Tools & Component Mapping Note.pdf](./Rag%20System%20%E2%80%93%20Tools%20&%20Component%20Mapping%20Note.pdf) — Technology stack and component selection

## Phase 1 Focus

- Document registry and metadata extraction
- Probe engine scaffolding
- Deterministic parsing stubs
- SQL lineage (PostgreSQL)

See `docs/USAGE.md` for quick commands.

## Philosophy

Trust-first, audit-driven ingestion with deterministic extraction and full lineage.
No ML hallucination, no blind parsing — every decision is reviewable.

## 5-Phase Implementation Plan

1. **Phase 1 (Current)**: Document Registry — Accept PDFs, extract metadata, store in SQL with immutable document_id
2. **Phase 2**: Probe Engine — Sample pages, detect structure (text-heavy, table-heavy, scanned), compute complexity
3. **Phase 3**: Deterministic Parsing — Extract text/tables, record method and confidence, populate blocks table
4. **Phase 4**: SQL Verification — Audit extracted data, ensure lineage, cross-check with source
5. **Phase 5 (Optional)**: Vector/Retrieval Hooks — Prepare for embeddings and LLM-based retrieval
