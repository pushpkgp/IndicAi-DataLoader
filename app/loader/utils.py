from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])