import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Class names
class_names = [
    'asteroid',
    'black_hole',
    'earth',
    'galaxy',
    'jupiter',
    'mars',
    'mercury',
    'neptune',
    'pluto',
    'saturn',
    'uranus',
    'venus'
]

# Load model
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    12
)

model.load_state_dict(
    torch.load("models/best_model.pth")
)

model.eval()

# Select image
image_path = r"C:\Users\Manasa\OneDrive\Desktop\SpaceObjectClassification\dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\test\saturn\saturn (184).jpg"

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")

rgb_img = np.array(image.resize((224,224))) / 255.0

input_tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():

    outputs = model(input_tensor)

    predicted_class = outputs.argmax(dim=1).item()

print(
    "Prediction:",
    class_names[predicted_class]
)

# Last convolution layer
target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(predicted_class)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.show()