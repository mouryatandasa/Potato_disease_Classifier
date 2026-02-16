from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    predicted_class: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)