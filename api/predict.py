from fastapi import APIRouter, UploadFile, File
from inference.inference_engine_y11 import run_inference
import shutil
import uuid
import os

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    file_id = f"{uuid.uuid4()}.jpg"
    file_path = f"uploads/{file_id}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_inference(file_path)

    return result
