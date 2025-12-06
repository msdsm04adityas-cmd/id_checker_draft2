from db.database import SessionLocal
from db import crud

db = SessionLocal()
crud.create_user(db, "admin", "admin123")

print("Admin user created: admin / admin123")


