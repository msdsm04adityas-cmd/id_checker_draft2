from fastapi import APIRouter, UploadFile, File
import shutil
import uuid
import os
from inference.inference_engine_y11 import run_inference

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save uploaded file
    file_id = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run inference
    result = run_inference(file_path)

    return {"status": "success", "result": result}
