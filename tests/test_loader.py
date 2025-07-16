from app.service.loader import StreamingDatasetLoader
from torchvision import transforms
from torch.utils.data import DataLoader

# Define basic image transform
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Initialize dataset
dataset = StreamingDatasetLoader(
    index_path='data/metadata/train.csv',
    split='train',
    modality='image',
    transform=image_transforms,
    shuffle=False
)

# Use PyTorch DataLoader
loader = DataLoader(dataset, batch_size=2, num_workers=0)

# Iterate and print
for batch in loader:
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")