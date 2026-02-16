from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  
import numpy as np 
from PIL import Image
import io
import uvicorn
import os
import logging

# App Initialization

app = FastAPI(
    title="Potato Disease Classifier API",
    description="Classifies potato leaf images",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Root Endpoint

@app.get("/")
async def root():
    return {
        "message": "Potato Disease Classifier API",
        "docs": "/docs",
        "endpoint": "/predict"
    }
# Load Model (lazy)

logger = logging.getLogger(__name__)
model = None
MODEL_PATH = "saved_model/potato_disease_model.keras"
DATA_DIR = "dataset/PlantVillage"

try:
    CLASS_NAMES = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
except Exception:
    CLASS_NAMES = []

IMG_SIZE = 224


def try_load_model():
    global model
    if model is not None:
        return model
    try:
        import tensorflow as tf
    except Exception as e:
        logger.exception("TensorFlow import failed: %s", e)
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
        return model
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        model = None
        return None
#image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# API Endpoint

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    mdl = try_load_model()
    if mdl is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not available. Check server logs for details.")

    predictions = mdl.predict(image)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    label = CLASS_NAMES[class_index] if CLASS_NAMES and class_index < len(CLASS_NAMES) else str(class_index)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



