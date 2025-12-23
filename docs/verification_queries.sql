-- Verify documents table has an entry for a given document_id
SELECT * FROM documents WHERE document_id = '<document_id>';

-- Count pages recorded for a document
SELECT COUNT(*) FROM pages WHERE document_id = '<document_id>';
