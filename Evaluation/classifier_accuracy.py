import os
import json
import cv2
from ultralytics import YOLO

CONFIG = "config.json"
TEST_DIR = "test_images"    # root folder

def load_config():
    with open(CONFIG, "r") as f:
        return json.load(f)


# ------------------------------
# NORMALIZE TRUE LABELS
# ------------------------------
def true_label_from_folder(path):
    folder = os.path.basename(os.path.dirname(path)).lower()

    mapping = {
        "aadhar": "aadhar_front",
        "aadhaar": "aadhar_front",
        "aadhaar_front": "aadhar_front",
        "aadhar_front": "aadhar_front",

        "pan": "pan_card_front",
        "pancard": "pan_card_front",
        "pan_card_front": "pan_card_front",

        "dl": "driving_license_front",
        "driving": "driving_license_front",
        "driving_license": "driving_license_front",
        "driving_license_front": "driving_license_front",
    }

    return mapping.get(folder, folder)


# ------------------------------
# YOLO SAFE PREDICT (PREVENT CRASH)
# ------------------------------
def safe_predict(model, img_path):
    """
    Prevent YOLO from crashing on unreadable/corrupt files.
    Returns: prediction result OR None
    """
    try:
        # Try to read image first
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read image → {img_path}")
            return None

        out = model(img_path)
        if not out or len(out) == 0:
            print(f"⚠️ YOLO returned empty result → {img_path}")
            return None

        return out[0]

    except Exception as e:
        print(f"❌ YOLO error on: {img_path}")
        print("   →", str(e))
        return None


# ------------------------------
# COMPUTE ACCURACY
# ------------------------------
def compute_accuracy():
    cfg = load_config()
    cls_model_path = cfg["models"]["Id_Classifier"]["path"]

    print(f"\n🔵 Loading classifier: {cls_model_path}")
    model = YOLO(cls_model_path)

    total = 0
    correct = 0
    wrong_samples = []

    print("\n📌 Starting evaluation...\n")

    for root, _, files in os.walk(TEST_DIR):
        for f in files:

            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"⏭ Skipping non-image file: {f}")
                continue

            img_path = os.path.join(root, f)
            total += 1

            true = true_label_from_folder(img_path)
            pred = None

            # Safe YOLO prediction
            result = safe_predict(model, img_path)
            if result is None:
                print(f"⛔ Skipped (invalid or unreadable): {img_path}")
                continue

            # Get prediction
            pred_idx = int(result.probs.top1)
            pred = result.names[pred_idx].lower()

            # Log
            print(f"{img_path} → TRUE={true} | PRED={pred}")

            if pred == true:
                correct += 1
            else:
                wrong_samples.append({
                    "path": img_path,
                    "true": true,
                    "pred": pred
                })

    # FINAL ACCURACY
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n-----------------------------")
    print(f"TOTAL IMAGES : {total}")
    print(f"CORRECT      : {correct}")
    print(f"WRONG        : {len(wrong_samples)}")
    print(f"ACCURACY     : {accuracy:.2f}%")
    print("-----------------------------")

    # Save wrong predictions
    with open("classifier_misclassified.json", "w") as f:
        json.dump(wrong_samples, f, indent=4)

    print("\n❗ Misclassified saved → classifier_misclassified.json")


if __name__ == "__main__":
    compute_accuracy()
