from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..config import POSTGRES_DSN

engine = create_engine(POSTGRES_DSN, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_session():
    return SessionLocal()
