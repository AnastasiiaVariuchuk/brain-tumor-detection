from roboflow import Roboflow
import supervision as sv
import numpy as np
import cv2

IMAGE_PATH = "b.jpg"     # заміни на свій шлях
TARGET_CLASS = "tumor"            # можна писати "tumor", "Tumor", "tumor-cell" тощо

# 1) Модель
rf = Roboflow(api_key="162uSH7JuhRNxACBn73k")
print("loading Roboflow workspace...")
project = rf.workspace().project("brain-tumor-8twd6")
print("loading Roboflow project...")
model = project.version(1).model

# 2) Інференс
result = model.predict(IMAGE_PATH, confidence=40, overlap=30).json()
preds = result.get("predictions", [])
if not preds:
    print("No predictions.")
    raise SystemExit(0)

# 3) Detections (API 0.26+)
detections = sv.Detections.from_inference(result)
print("Total detections:", len(detections))

# 4) Класи та фільтр (case-insensitive, частковий збіг)
classes = [p["class"] for p in preds]
print("Classes in result:", sorted(set(classes)))

tc = TARGET_CLASS.lower()
mask = np.array([tc in c.lower() for c in classes])  # 'tumor' матчиться на 'Tumor-Cell'
detections_f = detections[mask]
preds_f = [p for p, keep in zip(preds, mask) if keep]

print(f"Detections for '{TARGET_CLASS}':", len(detections_f))

# 5) Анотація зображення
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Не знайдено файл: {IMAGE_PATH}")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Підписи: Class + confidence%
labels = [f"{p['class']} {p.get('confidence', 0)*100:.0f}%" for p in preds_f]

annotated = box_annotator.annotate(scene=image, detections=detections_f)
annotated = label_annotator.annotate(scene=annotated, detections=detections_f, labels=labels)

sv.plot_image(image=annotated, size=(16, 16))
