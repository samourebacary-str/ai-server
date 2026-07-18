from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Autoriser les appels depuis n'importe quel domaine (votre frontend Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle YOLOv8 nano (léger et rapide)
model = YOLO("yolov8n.pt")

# Classes YOLO standard (COCO dataset) qu'on considère "interdites" pendant un examen
# Référez-vous à la liste des 80 classes COCO : https://docs.ultralytics.com/datasets/detect/coco/
SUSPICIOUS_CLASSES = {
    "cell phone": "ai_phone",
    "book": "ai_book",
    "laptop": "ai_laptop",
    "tablet": "ai_tablet",
    "remote": "ai_remote",
    "person": "ai_multiple_persons",  # géré séparément (compte le nombre)
}

CONFIDENCE_THRESHOLD = 0.30


@app.get("/")
def root():
    return {"status": "YOLOv8 detection server running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"detected": "none", "confidence": 0}

        results = model(img, verbose=False)

        person_count = 0
        best_detection = None
        best_confidence = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                if cls_name == "person":
                    person_count += 1
                    continue

                if cls_name in SUSPICIOUS_CLASSES and confidence > best_confidence:
                    best_detection = SUSPICIOUS_CLASSES[cls_name]
                    best_confidence = confidence

        # Priorité : plusieurs personnes détectées
        if person_count > 1:
            return {
                "detected": "multiple_persons",
                "confidence": 1.0,
                "person_count": person_count
            }

        # Priorité : aucune personne détectée (étudiant absent du cadre)
        if person_count == 0:
            return {"detected": "no_face", "confidence": 1.0}

        if best_detection:
            return {
                "detected": best_detection.replace("ai_", ""),
                "confidence": round(best_confidence, 2)
            }

        return {"detected": "none", "confidence": 0}

    except Exception as e:
        return {"detected": "none", "confidence": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
