import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Attempt to load environment variables from cfg/.env
try:
    from dotenv import load_dotenv
    # Build path: current_file_dir/../../cfg/.env
    env_path = Path(__file__).resolve().parent.parent.parent / "cfg" / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Default connection string - update as needed or set DATABASE_URL env var
DATABASE_URL = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    return SessionLocal()