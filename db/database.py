from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# LOAD .env FIRST
load_dotenv()

# NOW READ DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")
print("LOADED URL:", DATABASE_URL)

# CREATE ENGINE
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
