from fastapi import FastAPI
from app.api.loader import router as loader_router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataloader")

# Create FastAPI app instance
app = FastAPI(
    title="DataLoader API",
    description="A streaming multimodal dataset loader for images, text, audio, and video",
    version="1.0.0"
)


# Health check route
@app.get("/")
def read_root():
    return {"message": "Welcome to the DataLoader API!"}


# Include loader routes under /api
app.include_router(loader_router, prefix="/api")
