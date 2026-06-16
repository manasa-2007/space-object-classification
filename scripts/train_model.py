import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Dataset paths
train_dir = r"dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\training"
val_dir = r"dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\validation"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    12
)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...\n")

# ---------- TRAINING ----------
model.train()

running_loss = 0
correct = 0
total = 0

for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    running_loss += loss.item()

    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

train_accuracy = 100 * correct / total

# ---------- VALIDATION ----------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total

print("\nTraining Completed!")
print(f"Average Loss: {running_loss / len(train_loader):.4f}")
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Validation Accuracy: {val_accuracy:.2f}%")
# Save model
torch.save(model.state_dict(), "models/best_model.pth")

print("\nModel Saved Successfully!")