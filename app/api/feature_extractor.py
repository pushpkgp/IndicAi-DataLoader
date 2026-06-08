from fastapi import APIRouter, Body

from app.service.feature.preprocessor import extract_1

router = APIRouter()

@router.post("/features")
async def extract_features(
        metadata_base_dir: str = Body(..., embed=True, description="Path to the index CSV file")
):
    await extract_1(metadata_file_path=metadata_base_dir)