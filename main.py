import os
import uuid
import shutil
import cv2
import hashlib

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# DB IMPORTS
from db.database import SessionLocal, Base, engine
from db import crud
from db import models

# Create Tables
Base.metadata.create_all(bind=engine)

# ML ENGINE
from inference.inference_engine_y11 import run_inference

# AUTH ROUTES
from auth.login import router as login_router

# API ROUTES
from api.predict import router as predict_router


# ---------------------------------------------------------
# IMAGE HASH FUNCTION (Duplicate Check)
# ---------------------------------------------------------
def get_image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="GUIDONA ID VALIDATOR")


# ---------------------------------------------------------
# STATIC FILES + TEMPLATES
# ---------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------
# CORS (optional)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------
app.include_router(login_router)
app.include_router(predict_router, prefix="/api")


# CLEAR EXTRACTED FIELDS
@app.post("/dashboard/clear-fields")
async def clear_fields(request: Request):
    request.app.state.last_result = None
    return RedirectResponse("/dashboard?refresh=1", status_code=302)


# CLEAR PROCESSED IMAGE
@app.post("/dashboard/clear-image")
async def clear_image(request: Request):
    request.app.state.last_image = None
    return RedirectResponse("/dashboard?refresh=1", status_code=302)


# ---------------------------------------------------------
# DASHBOARD (GET)
# ---------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if request.cookies.get("auth") != "yes":
        return RedirectResponse("/login")

    result = getattr(request.app.state, "last_result", None)
    image_path = getattr(request.app.state, "last_image", None)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image_path": image_path
        }
    )

@app.post("/dashboard/clear-all")
async def clear_all(request: Request):
    request.app.state.last_result = None
    request.app.state.last_image = None
    return RedirectResponse("/dashboard?refresh=1", status_code=302)


# ---------------------------------------------------------
# DASHBOARD UPLOAD (POST → REDIRECT → GET)
# ---------------------------------------------------------
@app.post("/dashboard/upload")
async def dashboard_upload(request: Request, file: UploadFile = File(...)):

    # AUTH CHECK
    if request.cookies.get("auth") != "yes":
        return RedirectResponse("/login")

    user_id = request.cookies.get("user_id")
    if user_id is None:
        return RedirectResponse("/login")

    user_id = int(user_id)
    db = SessionLocal()

    # Ensure uploads folder exists
    os.makedirs("uploads", exist_ok=True)

    # Save uploaded file
    file_id = f"{uuid.uuid4()}.jpg"
    file_path = f"uploads/{file_id}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ------------------------------------
    # DUPLICATE DETECTION
    # ------------------------------------
    img_hash = get_image_hash(file_path)

    existing_doc = crud.get_document_by_hash(db, img_hash)
    if existing_doc:
        print("DUPLICATE DOCUMENT — Using previous processed result")

        # Load previous result
        request.app.state.last_result = {
            "document_type": existing_doc.doc_type,
            "fields_status": existing_doc.fields_status,
            "fields": {f.field_name: f.field_value for f in existing_doc.fields},
        }
        request.app.state.last_image = existing_doc.processed_image_path

        return RedirectResponse("/dashboard?refresh=1", status_code=302)

    # ------------------------------------
    # RUN INFERENCE FOR NEW DOCUMENT
    # ------------------------------------
    result = run_inference(file_path)

    # Draw bounding boxes
    boxed_path = f"uploads/boxed_{file_id}"
    img = cv2.imread(file_path)

    if result.get("boxes"):
        for b in result["boxes"]:
            cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 150, 255), 2)
            cv2.putText(
                img,
                b["label"],
                (b["x1"], b["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 150, 255),
                2
            )

    cv2.imwrite(boxed_path, img)

    # SAFETY DEFAULTS
    fields = result.get("fields") or {}
    doc_type = result.get("document_type") or "unknown"
    status = result.get("fields_status") or "not_detected"

    # SAVE DOCUMENT IN DB
    doc = crud.save_document(
        db=db,
        user_id=user_id,
        original=file_path,
        processed=boxed_path,
        doc_type=doc_type,
        fields_status=status,
        image_hash=img_hash
    )

    # SAVE FIELDS
    crud.save_fields(db, doc.id, fields)

    # Update dashboard state
    request.app.state.last_result = result
    request.app.state.last_image = boxed_path

    return RedirectResponse("/dashboard?refresh=1", status_code=302)


# ---------------------------------------------------------
# DEFAULT ROOT → LOGIN PAGE
# ---------------------------------------------------------
@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse("/login")
