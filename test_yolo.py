from ultralytics import YOLO
import torch

print("GPU available:", torch.cuda.is_available())

for model_name in ["yolo11n.pt", "yolo11s.pt"]:
    print(f"\nTesting {model_name}")

    model = YOLO(model_name)
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=0,
        imgsz=640,
        save=True
    )

    print("Prediction complete")