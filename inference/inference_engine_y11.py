import os
os.environ["ULTRALYTICS_IGNORE_GIT"] = "True"

from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import json
import re

# -------------------------------------------------
# LOAD CONFIG
# -------------------------------------------------
with open(r"C:\Users\ABC\Desktop\id_checker\inference\config.json") as f:
    CFG = json.load(f)

# -------------------------------------------------
# GLOBAL OCR + MODELS
# -------------------------------------------------
ocr = PaddleOCR(lang="en")

# Classifier model (load once)
CLS_MODEL = YOLO(r"C:\Users\ABC\Desktop\id_checker\models\Id_Classifier.pt")

# Detection models cache
DET_MODELS = {}  # {det_key: YOLO(model_path)}

# Blur threshold (you can tune)
BLUR_THRESHOLD = 80.0

# Text length threshold to treat as "random document"
LONG_TEXT_THRESHOLD = 300

# Doc types jinke liye face zaroori nahi
NO_FACE_ALLOWED = {
    "pan_card_front",
    "voterid_front",
    "aadhaar_back",
    "pan_back",
    "dl_back",
    "driving_license_back",
    "voterid_back"
}


# -------------------------------------------------
# HELPER: BLUR DETECTION
# -------------------------------------------------
def is_blurry(img, threshold: float = BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, float(fm)


# -------------------------------------------------
# HELPER: FULL-IMAGE OCR → TEXT STRING
# -------------------------------------------------
def ocr_full_image(img):
    ocr_out = ocr.ocr(img)
    if not ocr_out:
        return "", []
    text = " ".join(w[1][0] for line in ocr_out for w in line)
    return text, ocr_out


# -------------------------------------------------
# HELPER: AADHAAR BACK ADDRESS EXTRACTION
# -------------------------------------------------
def extract_aadhaar_address_from_text(full_text: str) -> str:
    """
    Very simple heuristic: Aadhaar back address usually contains 'Address'
    and multiple commas / line breaks. We try to capture from 'Address'
    to the end.
    """
    lower = full_text.lower()
    idx = lower.find("address")
    if idx == -1:
        return ""

    addr = full_text[idx:]
    # Thoda clean karo
    addr = addr.replace("Address", "").replace("ADDRESS", "").strip()
    # Extra spaces compress
    addr = re.sub(r"\s{2,}", " ", addr)
    return addr


# -------------------------------------------------
# HELPER: DL VALIDITY EXTRACTION
# -------------------------------------------------
def extract_dl_validity_from_text(full_text: str) -> str:
    """
    Search for patterns like 'Valid Till 12-08-2030' or 'Valid up to 2030-08-12'
    """
    lower = full_text.lower()

    # Common phrases
    if "valid till" in lower or "valid upto" in lower or "valid up to" in lower:
        # Extract near that phrase (window)
        for phrase in ["valid till", "valid upto", "valid up to"]:
            idx = lower.find(phrase)
            if idx != -1:
                window = full_text[idx: idx + 50]
                # Find date-like patterns
                m = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", window)
                if m:
                    return m.group(1)

    # Fallback: any date-like pattern (last resort)
    m2 = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", full_text)
    if m2:
        return m2.group(1)

    return ""


# -------------------------------------------------
# HELPER: PAN BACK SIGNATURE PRESENCE
# -------------------------------------------------
def detect_pan_signature_presence(full_text: str) -> bool:
    """
    Just check if word 'signature' or 'sig.' appears.
    For real production you'd use a detector on signature ROI.
    """
    lower = full_text.lower()
    keywords = ["signature", "sig."]
    return any(k in lower for k in keywords)


# -------------------------------------------------
# HELPER: BASIC FRAUD FLAGS
# -------------------------------------------------
def compute_fraud_flags(
    doc_type: str,
    has_face: bool,
    conf: float,
    is_blur: bool,
    blur_score: float,
    full_text: str
):
    flags = []

    # Low confidence
    if conf < 0.60:
        flags.append(f"Low classifier confidence ({conf:.2f})")

    # Expected face but missing
    if doc_type in ["aadhaar_front", "voterid_front", "driving_license_front"] and not has_face:
        flags.append("No face detected on front side ID")

    # Very blurry
    if is_blur:
        flags.append(f"Image is blurry (Laplacian variance={blur_score:.1f})")

    # Too little text for ID
    if len(full_text.strip()) < 20:
        flags.append("Very low text content – may not be an ID card")

    # Example: PAN number format sanity (simple)
    if "pan" in doc_type:
        pan_match = re.search(r"[A-Z]{5}\d{4}[A-Z]", full_text)
        if not pan_match:
            flags.append("PAN format not detected in text")

    return flags


# -------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------
def run_inference(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return {
            "document_type": "others",
            "fields_status": "not_detected",
            "fields": {},
            "boxes": [],
            "quality": {"is_blurry": True, "blur_score": 0.0},
            "fraud": {"flags": ["Image not readable"], "is_potentially_fraud": True}
        }

    # -------------------------------------------------
    # 0. QUALITY CHECKS (BLUR + FACE)
    # -------------------------------------------------
    is_blur, blur_score = is_blurry(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    has_face = len(faces) > 0

    # -------------------------------------------------
    # 1. CLASSIFIER
    # -------------------------------------------------
    cls_output = CLS_MODEL(image_path)[0]

    conf = float(cls_output.probs.top1conf)
    pred_label = cls_output.names[int(cls_output.probs.top1)]

    print("Classifier Prediction:", pred_label, "| Confidence:", conf)

    # ---------------- SMART DECISION RULE -----------------
    # Rule 1: Low confidence
    if conf < 0.75:
        doc_type = "others"

    # Rule 2: No face → Not ID card (except certain types)
    elif (not has_face) and (pred_label not in NO_FACE_ALLOWED):
        doc_type = "others"

    # Rule 3: Too much text → Not ID card
    else:
        full_text, _ = ocr_full_image(img)
        if len(full_text) > LONG_TEXT_THRESHOLD:
            doc_type = "others"
        else:
            doc_type = pred_label

    print("Final Document Type:", doc_type)

    # -------------------------------------------------
    # 2. HANDLE UNKNOWN DOC
    # -------------------------------------------------
    if doc_type == "others":
        full_text, _ = ocr_full_image(img)
        flags = compute_fraud_flags(
            doc_type=doc_type,
            has_face=has_face,
            conf=conf,
            is_blur=is_blur,
            blur_score=blur_score,
            full_text=full_text,
        )
        return {
            "document_type": "others",
            "fields_status": "detected" if full_text else "not_detected",
            "fields": {"Raw_Text": full_text} if full_text else {},
            "boxes": [],
            "quality": {
                "is_blurry": is_blur,
                "blur_score": blur_score
            },
            "fraud": {
                "flags": flags,
                "is_potentially_fraud": len(flags) > 0
            }
        }

    # -------------------------------------------------
    # 3. DETECTOR MODEL LOAD (CACHED)
    # -------------------------------------------------
    det_key = CFG["doc_type_to_model"][doc_type]
    if det_key not in DET_MODELS:
        det_model_path = r"C:\Users\ABC\Desktop\id_checker\\" + CFG["models"][det_key]["path"]
        DET_MODELS[det_key] = YOLO(det_model_path)

    det_model = DET_MODELS[det_key]
    det_output = det_model(image_path)[0]

    results = {}
    box_list = []

    # -------------------------------------------------
    # 4. FIELD EXTRACTION (BOX-WISE OCR)
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

        # Store clean box
        box_list.append({
            "label": label,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    # -------------------------------------------------
    # 4.5 – FULL TEXT FOR EXTRA LOGIC & FRAUD
    # -------------------------------------------------
    full_text, _ = ocr_full_image(img)

    # Aadhaar back → address
    if doc_type == "aadhaar_back":
        addr = extract_aadhaar_address_from_text(full_text)
        if addr:
            results["Address"] = addr

    # DL front/back → validity
    if "driving" in doc_type or "dl_" in doc_type:
        validity = extract_dl_validity_from_text(full_text)
        if validity:
            results["Valid_Till"] = validity

    # PAN back → signature presence
    if "pan" in doc_type and "back" in doc_type:
        has_signature = detect_pan_signature_presence(full_text)
        results["Signature_Present"] = "Yes" if has_signature else "No"

    # -------------------------------------------------
    # 5. NO FIELDS CASE
    # -------------------------------------------------
    if len(results) == 0:
        flags = compute_fraud_flags(
            doc_type=doc_type,
            has_face=has_face,
            conf=conf,
            is_blur=is_blur,
            blur_score=blur_score,
            full_text=full_text,
        )
        return {
            "document_type": doc_type,
            "fields_status": "not_detected",
            "fields": {},
            "boxes": [],
            "quality": {
                "is_blurry": is_blur,
                "blur_score": blur_score
            },
            "fraud": {
                "flags": flags,
                "is_potentially_fraud": len(flags) > 0
            }
        }

    # -------------------------------------------------
    # 6. SUCCESS RETURN
    # -------------------------------------------------
    fraud_flags = compute_fraud_flags(
        doc_type=doc_type,
        has_face=has_face,
        conf=conf,
        is_blur=is_blur,
        blur_score=blur_score,
        full_text=full_text,
    )

    return {
        "document_type": doc_type,
        "fields_status": "detected",
        "fields": results,
        "boxes": box_list,
        "quality": {
            "is_blurry": is_blur,
            "blur_score": blur_score
        },
        "fraud": {
            "flags": fraud_flags,
            "is_potentially_fraud": len(fraud_flags) > 0
        }
    }


# -------------------------------------------------
# LOCAL TEST (optional)
# -------------------------------------------------
if __name__ == "__main__":
    test_img = r"C:\Users\ABC\Desktop\id_checker\test_images\aadhar\test.jpg"
    out = run_inference(test_img)
    print(json.dumps(out, indent=2, ensure_ascii=False))
