import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset paths
train_dir = r"dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\training"
val_dir = r"dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\validation"
test_dir = r"dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\test"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Get first batch
images, labels = next(iter(train_loader))

print("Image Batch Shape:")
print(images.shape)

print("\nLabel Batch Shape:")
print(labels.shape)

print("\nFirst 10 Labels:")
print(labels[:10])