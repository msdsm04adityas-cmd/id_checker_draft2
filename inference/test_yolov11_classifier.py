from ultralytics import YOLO

print("Loading YOLOv11 model...")
model = YOLO(r"C:\Users\ABC\Desktop\id_checker\models\Id_Classifier.pt")
print("Model Loaded Successfully!")

result = model(r"C:\Users\ABC\Desktop\id_checker\test_images\aadhar\test.jpg")[0]
print("Predicted:", result.names[result.probs.top1])
