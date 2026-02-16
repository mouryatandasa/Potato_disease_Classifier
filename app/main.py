from fastapi import FastAPI
from sqlmodel import SQLModel
from app.routers import predict
from app.app import root
from app.database import engine
app = FastAPI(title="Potato Disease Classifier")

SQLModel.metadata.create_all(engine)

app.include_router(predict.router)
@app.get("/")
async def root():
    return {
        "message": "Potato Disease Classifier API",
        "docs": "/docs",
        "endpoint": "/predict"
    }