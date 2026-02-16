import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
from src.data_loader import get_data
from src.model import build_cnn
from src.utils import plot_history

DATA_DIR = "dataset/PlantVillage"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "saved_model/potato_disease_model.keras"

def train():
    train_ds, val_ds = get_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_ds.class_names)

    model = build_cnn(num_classes, IMG_SIZE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    plot_history(history)

if __name__ == "__main__":
    train()