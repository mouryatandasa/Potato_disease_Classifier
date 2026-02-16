from sqlmodel import Session
from app.database import engine
from app.model import Prediction

def save_prediction(filename: str, predicted_class: str, confidence: float):
    with Session(engine) as session:
        prediction = Prediction(
            filename=filename,
            predicted_class=predicted_class,
            confidence=confidence
        )
        session.add(prediction)
        session.commit()