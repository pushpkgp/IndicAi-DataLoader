import logging

import pandas as pd
from fastapi import FastAPI
from app.api.feature_extractor import router as loader_router

# Create FastAPI app instance
app = FastAPI(
    title="DataLoader API",
    description="A streaming multimodal dataset loader for images, text, audio, and video",
    version="1.0.0"
)

pd.options.mode.copy_on_write = True

logger = logging.getLogger("feature_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Health check route
@app.get("/")
def read_root():
    return {"message": "Welcome to the DataLoader API!"}

# Include loader routes under /api
app.include_router(loader_router, prefix="/api")
