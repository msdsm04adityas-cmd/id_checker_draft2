from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password_hash = Column(Text)
    role = Column(String(20), default="user")
    created_at = Column(TIMESTAMP, server_default=func.now())



class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    original_image_path = Column(Text)
    processed_image_path = Column(Text)
    doc_type = Column(String(50))
    fields_status = Column(String(20))
    image_hash = Column(String(64), unique=True)   # NEW
    created_at = Column(TIMESTAMP, server_default=func.now())

    fields = relationship("ExtractedField", back_populates="document")


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    field_name = Column(String(100))
    field_value = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

    document = relationship("Document", back_populates="fields")
