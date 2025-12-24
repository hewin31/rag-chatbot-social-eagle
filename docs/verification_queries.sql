-- Phase 4: SQL Verification Queries

-- 1. Document overview: filename, status, page count
SELECT 
  document_id, 
  filename, 
  ingestion_status, 
  page_count, 
  file_size_bytes,
  created_at
FROM documents
ORDER BY created_at DESC;

-- 2. Block statistics for a document
SELECT 
  document_id,
  block_type,
  COUNT(*) as count,
  AVG(confidence) as avg_confidence,
  MIN(confidence) as min_confidence,
  MAX(confidence) as max_confidence
FROM blocks
WHERE document_id = '<document_id>'
GROUP BY document_id, block_type;

-- 3. Extraction method breakdown
SELECT 
  document_id,
  extraction_method,
  COUNT(*) as count
FROM blocks
WHERE document_id = '<document_id>'
GROUP BY document_id, extraction_method;

-- 4. Low confidence blocks (< 50)
SELECT 
  document_id,
  page_number,
  block_type,
  confidence,
  LENGTH(content) as content_length
FROM blocks
WHERE document_id = '<document_id>'
  AND confidence < 50
ORDER BY confidence ASC;

-- 5. Missing page numbers (if expecting sequential pages)
WITH page_nums AS (
  SELECT DISTINCT page_number FROM blocks 
  WHERE document_id = '<document_id>'
)
SELECT 
  generate_series(
    (SELECT MIN(page_number) FROM page_nums),
    (SELECT MAX(page_number) FROM page_nums)
  ) as expected_page
WHERE generate_series NOT IN (SELECT page_number FROM page_nums);

-- 6. Lineage: document -> blocks with full context
SELECT 
  d.document_id,
  d.filename,
  b.page_number,
  b.block_type,
  b.extraction_method,
  b.confidence,
  LENGTH(b.content) as content_length,
  b.id as block_id
FROM documents d
LEFT JOIN blocks b ON d.document_id = b.document_id
WHERE d.document_id = '<document_id>'
ORDER BY b.page_number, b.id;

-- 7. Sample block content (first 500 chars) for manual review
SELECT 
  page_number,
  block_type,
  confidence,
  SUBSTRING(content, 1, 500) as content_preview
FROM blocks
WHERE document_id = '<document_id>'
LIMIT 10;

-- 8. Completeness check: total blocks per page
SELECT 
  page_number,
  COUNT(*) as block_count,
  SUM(LENGTH(content)) as total_content_bytes
FROM blocks
WHERE document_id = '<document_id>'
GROUP BY page_number
ORDER BY page_number;
