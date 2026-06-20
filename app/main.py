import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from app.api.feature_extractor import router as loader_router
from app.service.feature.image.model_registry import get_segmentation_assets

pd.options.mode.copy_on_write = True

logger = logging.getLogger("feature_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DataLoader API")

    try:
        model, postprocess_params = get_segmentation_assets()

        app.state.segmentation_model = model
        app.state.postprocess_params = postprocess_params

        logger.info("Segmentation assets loaded successfully")

    except Exception as exc:
        logger.error("Failed to load segmentation assets: %s", exc)
        raise

    yield

    logger.info("Shutting down DataLoader API")

app = FastAPI(
    title="DataLoader API",
    description="A streaming multimodal dataset loader for images, text, audio, and video",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the DataLoader API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

app.include_router(loader_router, prefix="/api")