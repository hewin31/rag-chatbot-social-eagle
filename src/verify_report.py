#!/usr/bin/env python3
import json
from src.verify.chunk_checks import verify_chunks

# -----------------------------
# CONFIG: set your document_id here
# -----------------------------
DOCUMENT_ID = "c5c20cb9-0cfe-424e-ad81-1f288363e7ae"

# -----------------------------
# Load the report
# -----------------------------
report = verify_chunks(DOCUMENT_ID)

# -----------------------------
# Print Chunks per block (text bar)
# -----------------------------
print("\nChunks per block:")
for block, count in report.get("chunks_per_block", {}).items():
    print(f"Block {block}: {'#'*count} ({count})")

# -----------------------------
# Print Token stats
# -----------------------------
tokens = report.get("token_stats", {})
print("\nToken stats:")
print(f"  min: {tokens.get('min')}")
print(f"  max: {tokens.get('max')}")
print(f"  avg: {tokens.get('avg')}")

# -----------------------------
# Print overall status
# -----------------------------
print("\nCreation methods:")
for method, val in report.get("creation_methods", {}).items():
    print(f"  {method}: {val}")

print(f"\nIssues found: {len(report.get('issues', []))}")
print(f"Overall report: {report.get('overall')}")
