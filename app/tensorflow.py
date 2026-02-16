import tensorflow as tf
import keras
model = tf.keras.models.load_model("saved_model/potato_disease_model.keras")
model = keras.models.load_model("saved_model/potato_disease_model.keras")