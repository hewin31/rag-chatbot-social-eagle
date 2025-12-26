from sqlalchemy import select, func
from src.db.session import get_session
from src.db.models import Entity

def debug_entities():
    session = get_session()
    try:
        # 1. Count total entities
        count = session.scalar(select(func.count(Entity.entity_id)))
        print(f"Total Entities in DB: {count}")

        # 2. List top 50 entities
        print("\n--- Top 50 Entities ---")
        stmt = select(Entity.entity_text, Entity.entity_type).limit(50)
        results = session.execute(stmt).all()
        for text, type_ in results:
            print(f"[{type_}] {text}")

        # 3. Check specific terms
        print("\n--- Checking for 'tax' and 'income' ---")
        for term in ["tax", "income", "penalty", "penalties"]:
            stmt = select(Entity).filter(Entity.entity_text.ilike(f"%{term}%"))
            matches = session.execute(stmt).scalars().all()
            if matches:
                print(f"Found '{term}': {[m.entity_text for m in matches]}")
            else:
                print(f"Did NOT find '{term}'")

    finally:
        session.close()

if __name__ == "__main__":
    debug_entities()
