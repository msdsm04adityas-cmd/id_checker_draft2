import os
os.environ["ULTRALYTICS_IGNORE_GIT"] = "True"

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import json

# Load config
with open(r"C:\Users\ABC\Desktop\id_checker\inference\config.json") as f:
    CFG = json.load(f)

# OCR
ocr = PaddleOCR(lang="en")


def run_inference(image_path):

    img = cv2.imread(image_path)

    # -------------------------------------------------
    # 0. FACE CHECK (Avoid random text pages)
    # -------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    has_face = len(faces) > 0

    # -------------------------------------------------
    # 1. CLASSIFIER
    # -------------------------------------------------
    cls_model = YOLO(r"C:\Users\ABC\Desktop\id_checker\models\Id_Classifier.pt")
    cls_output = cls_model(image_path)[0]

    conf = cls_output.probs.top1conf
    pred_label = cls_output.names[int(cls_output.probs.top1)]

    print("Classifier Prediction:", pred_label, "| Confidence:", conf)

    # ---------------- SMART DECISION RULE (UPDATED) -----------------

    BACK_CLASSES = [
        "pan_card_back",
        "aadhaar_back",
        "voterid_back",
        "driving_license_back"
    ]

    FRONT_CLASSES = [
        "pan_card_front",
        "aadhaar_front",
        "voterid_front",
        "driving_license_front"
    ]

    # RULE 1 — low confidence → others
    if conf < 0.60:
        doc_type = "others"

    # RULE 2 — FRONT classes must have face (PAN front optional)
    elif pred_label in FRONT_CLASSES and pred_label != "pan_card_front" and not has_face:
        doc_type = "others"

    # RULE 3 — BACKSIDE allowed long text
    else:
        ocr_out = ocr.ocr(img)
        text_full = ""
        if ocr_out:
            text_full = " ".join(w[1][0] for line in ocr_out for w in line)

        if len(text_full) > 350 and pred_label not in BACK_CLASSES:
            doc_type = "others"
        else:
            doc_type = pred_label

    print("Final Document Type:", doc_type)

    # -------------------------------------------------
    # 2. HANDLE UNKNOWN DOC
    # -------------------------------------------------
    if doc_type == "others":
        ocr_out = ocr.ocr(img)
        raw_text = ""
        if ocr_out:
            raw_text = " ".join(w[1][0] for line in ocr_out for w in line)

        return {
            "document_type": "others",
            "fields_status": "detected",
            "fields": {"Raw_Text": raw_text},
            "boxes": []
        }

    # -------------------------------------------------
    # 3. DETECTOR
    # -------------------------------------------------
    det_key = CFG["doc_type_to_model"][doc_type]
    det_model_path = r"C:\Users\ABC\Desktop\id_checker\\" + CFG["models"][det_key]["path"]

    det_model = YOLO(det_model_path)
    det_output = det_model(image_path)[0]

    results = {}
    box_list = []

    # -------------------------------------------------
    # 4. FIELD EXTRACTION
    # -------------------------------------------------
    for box in det_output.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        ocr_out = ocr.ocr(crop)
        text = ""
        if ocr_out:
            text = " ".join(w[1][0] for line in ocr_out for w in line)

        cls_id = int(box.cls[0])
        raw_label = det_output.names[cls_id]

        # PAN custom mapping
        pan_map = {
            0: "PAN Number",
            1: "Name",
            2: "Father Name",
            3: "Date of Birth",
            4: "Details"
        }

        if doc_type == "pan_card_front":
            label = pan_map.get(cls_id, raw_label)
        else:
            label = raw_label

        results[label] = text

        box_list.append({
            "label": label,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    # -------------------------------------------------
    # 5. NO FIELDS CASE
    # -------------------------------------------------
    if len(results) == 0:
        return {
            "document_type": doc_type,
            "fields_status": "not_detected",
            "fields": {},
            "boxes": []
        }

    # -------------------------------------------------
    # 6. SUCCESS RETURN
    # -------------------------------------------------
    return {
        "document_type": doc_type,
        "fields_status": "detected",
        "fields": results,
        "boxes": box_list
    }
