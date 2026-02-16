from sqlmodel import create_engine

DATABASE_URL = "sqlite:///predictions.db"
engine = create_engine(DATABASE_URL, echo=True)