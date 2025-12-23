# Usage

## Ingest (CLI)

Make sure to configure `cfg/.env.example` as `cfg/.env` and set `POSTGRES_DSN`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Ingest a PDF (example):

```bash
python -m src.ingest.ingest_cli ingest /path/to/file.pdf
```
