# FastAPI + PyTorch Multi-Modal Dataset API

## Features
- PyTorch model inference (dummy image classifier)
- FastAPI REST API for uploading and predicting
- Streaming, sharded data loader for large-scale training
- Docker-compatible

## Endpoints
- `GET /` - Health check
- `POST /api/predict/` - Upload image for prediction

## Usage
```bash
docker build -t fastapi-pytorch-app .
docker run -p 8000:8000 fastapi-pytorch-app
```

Or run locally:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```