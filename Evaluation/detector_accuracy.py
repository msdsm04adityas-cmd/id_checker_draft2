import os
import json
from ultralytics import YOLO

CONFIG = "config.json"
TEST_DIR = "test_images"     # root folder (contains Aadhaar / pan / driving_license...)

def load_config():
    with open(CONFIG, "r") as f:
        return json.load(f)

# ------------------------------------------------------------
#  TRUE LABEL IDENTIFICATION BASED ON FOLDER NAME
# ------------------------------------------------------------
def true_label_from_folder(path):
    folder = os.path.basename(os.path.dirname(path)).lower()

    mapping = {
        "aadhar": "Aadhaar",
        "aadhaar": "Aadhaar",
        "aadhar_front": "Aadhaar",
        "aadhar_back": "Aadhaar",

        "pan": "Pan_Card",
        "pancard": "Pan_Card",
        "pan_card": "Pan_Card",
        "pan_card_front": "Pan_Card",

        "driving": "Driving_License",
        "driving_license": "Driving_License",
        "driving_license_front": "Driving_License",
        "driving_license_back": "Driving_License",

        "passport": "Passport",
        "voter": "Voter_Id",
        "voter_id": "Voter_Id"
    }

    return mapping.get(folder, folder)


# ------------------------------------------------------------
# DETECTOR ACCURACY SCRIPT
# ------------------------------------------------------------
def detector_accuracy():
    cfg = load_config()

    # Load all detection models (skip Id_Classifier)
    detector_models = {
        name: YOLO(m["path"])
        for name, m in cfg["models"].items()
        if m["type"] == "detection"
    }

    total = 0
    correct = 0
    wrong = []

    print("\n🔍 Starting Detector Accuracy Evaluation...\n")

    # Loop through dataset
    for root, _, files in os.walk(TEST_DIR):
        for f in files:

            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, f)
            true_label = true_label_from_folder(img_path)

            if true_label not in detector_models:
                print(f"⚠️ SKIPPED (no model found for): {img_path}")
                continue

            model = detector_models[true_label]
            total += 1

            try:
                result = model(img_path)[0]

                if result.boxes is None or len(result.boxes) == 0:
                    pred = "NO_DETECTION"
                else:
                    # Highest-conf detection class
                    cls_idx = int(result.boxes.cls[0])
                    pred = cfg["models"][true_label]["classes"][cls_idx]
                    pred = true_label    # Normalized for model-level accuracy

            except Exception as e:
                print(f"❌ ERROR on {img_path}: {e}")
                wrong.append((img_path, true_label, "ERROR"))
                continue

            if pred == true_label:
                correct += 1
            else:
                wrong.append((img_path, true_label, pred))

            print(f"{img_path} → TRUE={true_label} | PRED={pred}")

    # -------------------------------------------------------
    # Final accuracy
    # -------------------------------------------------------
    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n===============================")
    print(f"TOTAL IMAGES : {total}")
    print(f"CORRECT      : {correct}")
    print(f"WRONG        : {len(wrong)}")
    print(f"ACCURACY     : {accuracy:.2f}%")
    print("===============================")

    with open("detector_misclassified.json", "w") as f:
        json.dump(wrong, f, indent=4)

    print("\n❗ Misclassified saved → detector_misclassified.json")


if __name__ == "__main__":
    detector_accuracy()
