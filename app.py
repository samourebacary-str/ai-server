from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# ⚠️ FIX CORS : Ajout de allow_credentials=True et explicitation des headers/méthodes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Charger le modèle YOLOv8 nano
model = YOLO("yolov8n.pt")

SUSPICIOUS_CLASSES = {
    "cell phone": "phone",
    "book": "book",
    "laptop": "laptop",
    "tablet": "tablet",
    "remote": "remote"
}

CONFIDENCE_THRESHOLD = 0.15


@app.get("/")
def root():
    return {"status": "YOLOv8 detection server running"}


@app.head("/")
def root_head():
    return {}


# ⚠️ FIX PREFLIGHT: Ajouter un endpoint OPTIONS explicite pour éviter les blocages CORS
@app.options("/detect")
async def detect_options():
    return {}


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

        # 1️⃣ PRIORITÉ ABSOLUE : Objet suspect
        if best_detection:
            return {
                "detected": best_detection,
                "confidence": round(best_confidence, 2)
            }

        # 2️⃣ PRIORITÉ SECONDAIRE : Plusieurs personnes
        if person_count > 1:
            return {
                "detected": "multiple_persons",
                "confidence": 1.0,
                "person_count": person_count
            }

        # 3️⃣ PRIORITÉ TERTIAIRE : Étudiant absent
        if person_count == 0:
            return {"detected": "no_face", "confidence": 1.0}

        # Si tout est OK
        return {"detected": "none", "confidence": 0}

    except Exception as e:
        return {"detected": "none", "confidence": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)