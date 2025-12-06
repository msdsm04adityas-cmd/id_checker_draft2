import os
from inference.inference_engine_y11 import run_inference

# YOUR TEST FOLDER PATH
TEST_FOLDER = r"C:\Users\ABC\Desktop\id_checker\test"

# Normalize mapping (test folder names → model names)
GT_MAP = {
    "aadhar": "aadhaar_front",
    "aadhaar": "aadhaar_front",
    "pan": "pan_card_front",
    "driving_license": "driving_license_front",
    "other": "others",
    "others": "others"
}

print("\n🔍 Checking test folder:", TEST_FOLDER)

if not os.path.exists(TEST_FOLDER):
    print("❌ ERROR: Test folder does NOT exist!")
    exit()

# Get class folders
classes = [d for d in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, d))]

print("\n📁 Found Classes:", classes)

total = 0
correct = 0
per_class = {cls: {"correct": 0, "total": 0} for cls in classes}

print("\n==============================")
print(" RUNNING TEST ACCURACY ")
print("==============================\n")

for gt_class in classes:

    folder_path = os.path.join(TEST_FOLDER, gt_class)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png",".jpeg"))]

    print(f"\n📂 Class '{gt_class}' → {len(images)} images")

    if len(images) == 0:
        print(f"⚠ WARNING: No images inside '{gt_class}' folder!")
        continue

    for img_name in images:

        img_path = os.path.join(folder_path, img_name)

        print(f"\n➡ Testing: {img_name} (GT: {gt_class})")

        try:
            result = run_inference(img_path)
            pred = result["document_type"]
        except Exception as e:
            print("❌ ERROR processing image:", e)
            continue

        # Normalize
        gt_label = GT_MAP.get(gt_class.lower(), gt_class.lower())
        pred_label = pred.lower().replace(" ", "_")

        print(f"   Predicted: {pred_label}")

        total += 1
        per_class[gt_class]["total"] += 1

        if pred_label == gt_label:
            correct += 1
            per_class[gt_class]["correct"] += 1

# -------------------------------
#   FINAL RESULTS
# -------------------------------

if total == 0:
    print("\n❌ NO IMAGES PROCESSED. CHECK FOLDER STRUCTURE.")
    exit()

accuracy = (correct / total) * 100

print("\n==============================")
print(f"🔥 FINAL OVERALL ACCURACY: {accuracy:.2f}%  ({correct}/{total})")
print("==============================\n")

print("📊 PER-CLASS ACCURACY:\n")
for cls, stats in per_class.items():
    if stats["total"] == 0:
        continue
    acc = (stats["correct"] / stats["total"]) * 100
    print(f"{cls:20s} → {acc:.2f}%  ({stats['correct']}/{stats['total']})")
