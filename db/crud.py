from sqlalchemy.orm import Session
from .models import User, Document, ExtractedField
from werkzeug.security  import generate_password_hash, check_password_hash


# -------- USERS --------
def create_user(db: Session, username: str, password: str):
    hashed = generate_password_hash(password)
    user = User(username=username, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not check_password_hash(user.password_hash, password):
        return None
    return user


# -------- DOCUMENT UPLOAD SAVE --------
def save_document(db: Session, user_id, original, processed, doc_type, fields_status, image_hash):
    doc = Document(
        user_id=user_id,
        original_image_path=original,
        processed_image_path=processed,
        doc_type=doc_type,
        fields_status=fields_status,
        image_hash=image_hash
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

# -------- DUPLICATE DOCUMENT CHECK --------
def get_document_by_hash(db: Session, image_hash: str):
    return db.query(Document).filter(Document.image_hash == image_hash).first()


# -------- FIELDS SAVE --------
def save_fields(db: Session, document_id, fields):
    for k, v in fields.items():
        entry = ExtractedField(
            document_id=document_id,
            field_name=k,
            field_value=v
        )
        db.add(entry)
    db.commit()


# -------- HISTORY --------
def get_all_documents(db: Session):
    return db.query(Document).order_by(Document.id.desc()).all()


def get_fields_for_document(db: Session, doc_id: int):
    return db.query(ExtractedField).filter(ExtractedField.document_id == doc_id).all()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()
