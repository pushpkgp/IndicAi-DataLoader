from fastapi import APIRouter, Body
from torch.utils.data import DataLoader

from app.config.logging_config import logger
from app.service.loader.loader import IndicDataLoader
from app.service.metadata_generator.metadata_generator import MetadataGenerator

router = APIRouter()

@router.post("/load/images")
def load_dataset(
        index_path: str = Body(..., embed=True, description="Path to the index CSV file"),
        split: str = Body("train", embed=True, description="Dataset split: train, val, or test"),
        modality: str = Body("image", embed=True, description="Data modality: image, text, audio, or videos")
):
    try:
        dataset = IndicDataLoader(
            index_path=index_path,
            split=split,
            modality=modality,
            shuffle=False
        )

        count = len(dataset.df)
        if count == 0:
            return {"message": "No valid records found after filtering and validation."}

        # Use PyTorch DataLoader
        loader = DataLoader(dataset, batch_size=2, num_workers=0)

        # Iterate and print
        for batch in loader:
            images, labels = batch

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"error": str(e)}

@router.post("/load/dataset/path")
def load_dataset_path(
        path: str = Body(..., embed=True, description="Path to the dataset")
):
        MetadataGenerator(path).generate_and_save_metadata()