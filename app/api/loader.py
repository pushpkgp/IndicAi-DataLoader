from fastapi import APIRouter, Body
from torch.utils.data import DataLoader

from app.config.logging_config import logger
from app.service.loader.loader import IndicDataLoader
from torchvision import transforms

router = APIRouter()

@router.post("/load/images")
def load_dataset(
        index_path: str = Body(..., embed=True, description="Path to the index CSV file"),
        split: str = Body("train", embed=True, description="Dataset split: train, val, or test"),
        modality: str = Body("image", embed=True, description="Data modality: image, text, audio, or videos")
):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]) if modality == 'image' else None

        dataset = IndicDataLoader(
            index_path=index_path,
            split=split,
            modality=modality,
            transform=transform,
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
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")

        # sample_batch = []
        # for i, item in enumerate(dataset):
        #     if i >= 3:
        #         break
        #     sample_batch.append(str(item[1]))  # Just return label info
        #
        # return {
        #     "message": f"Loaded {count} records.",
        #     "sample_labels": sample_batch
        # }

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"error": str(e)}
