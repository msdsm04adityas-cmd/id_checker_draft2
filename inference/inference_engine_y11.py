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

    # -------- CLASSIFIER --------
    cls_model = YOLO(r"C:\Users\ABC\Desktop\id_checker\models\Id_Classifier.pt")
    cls_output = cls_model(image_path)[0]

    doc_type = cls_output.names[cls_output.probs.top1]
    print("Document Type:", doc_type)

    # -------- HANDLE UNKNOWN DOC ("others") --------
    if doc_type == "others":
        full_img = cv2.imread(image_path)
        ocr_out = ocr.ocr(full_img)
        raw_text = " ".join([w[1][0] for line in ocr_out for w in line]) if ocr_out else ""

        return {
            "document_type": "others",
            "fields": {"Raw_Text": raw_text},
            "boxes": []
        }

    # -------- DETECTOR --------
    det_key = CFG["doc_type_to_model"][doc_type]
    det_model_path = r"C:\Users\ABC\Desktop\id_checker\\" + CFG["models"][det_key]["path"]

    det_model = YOLO(det_model_path)
    det_output = det_model(image_path)[0]

    img = cv2.imread(image_path)
    results = {}

    # -------- OCR FIELD EXTRACTION --------
    for box in det_output.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        ocr_out = ocr.ocr(crop)
        text = ""
        if ocr_out:
            text = " ".join(w[1][0] for line in ocr_out for w in line)

        cls_id = int(box.cls[0])
        raw_label = det_output.names[cls_id]

        # -------- PAN CUSTOM FIELD MAPPING --------
        pan_map = {
            "0": "PAN Number",
            "1": "Name",
            "2": "Father Name",
            "3": "Date of Birth",
            "4": "Details"
        }

        if doc_type == "pan_card_front":
            label = pan_map.get(str(cls_id), f"Field {cls_id}")
        else:
            label = raw_label

        results[label] = text

    # -------- CLEAN BOX LIST --------
    h, w = img.shape[:2]
    box_list = []

    for b in det_output.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cls_id = int(b.cls[0])
        label = det_output.names[cls_id]

        box_list.append({
            "label": label,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    # -------- IF NO FIELDS DETECTED --------
    if len(results) == 0:
        return {
            "fields_status": "not_detected",
            "fields": {},
            "boxes": []
        }

    # -------- OTHERWISE NORMAL RETURN --------
    return {
        "document_type": doc_type,
        "fields_status": "detected",
        "fields": results,
        "boxes": box_list
    }


# TEST
if __name__ == "__main__":
    out = run_inference(r"C:\Users\ABC\Desktop\id_checker\test_images\aadhar\test.jpg")
    print(out)
