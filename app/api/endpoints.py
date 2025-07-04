from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from app.models.dummy_model import DummyClassifier
import io

router = APIRouter()
model = DummyClassifier()
model.eval()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    predicted = outputs.argmax(dim=1).item()
    return {"predicted_class": predicted}