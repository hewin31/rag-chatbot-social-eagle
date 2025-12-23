from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:password@localhost:5432/rag_poc")
PDF_STORAGE_PATH = Path(os.getenv("PDF_STORAGE_PATH", "./data/pdfs")).expanduser()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def ensure_paths():
    PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
