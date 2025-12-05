import json

with open("config.json", "r") as f:
    cfg = json.load(f)

print("\nDETECTOR CLASS MAPPING:\n")

for model_name, m in cfg["models"].items():
    if "classes" in m:
        print(f"{model_name} → {m['classes']}")
