from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Path to your trained model
MODEL_PATH = "saved_model/potato_disease_model.keras"

# Load the model once
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model = None

# Load dataset just to get class order (not for training here)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/PlantVillage",
    image_size=(256, 256),
    batch_size=32
)
class_names = train_ds.class_names
print("CLASS ORDER:", class_names)

class PotatoInput(BaseModel):
    image_url: str

def preprocess_image(image_bytes):
    """Resize and normalize image for model input."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))  # match training size
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@router.post("/")
async def predict_disease(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )

    # Read and preprocess uploaded image
    contents = await file.read()
    input_data = preprocess_image(contents)

    # Run prediction
    preds = model.predict(input_data)
    predicted_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    # Use the actual training class order
    label = class_names[predicted_class]

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence, 4),
        "raw_scores": [round(float(x), 4) for x in preds[0]],
        "class_names": class_names
    }