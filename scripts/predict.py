import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class Names
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

# Explanations
explanations = {
    "asteroid": "The model identified irregular rocky structures and asteroid-like surface patterns.",

    "black_hole": "The model focused on the dark central region and surrounding accretion features typical of black holes.",

    "earth": "The model detected Earth's characteristic blue oceans, clouds, and continental structures.",

    "galaxy": "The model focused on the bright galactic core and spiral or clustered stellar structures.",

    "jupiter": "The model recognized Jupiter's large planetary body and atmospheric band patterns.",

    "mars": "The model detected the reddish surface coloration and planetary appearance of Mars.",

    "mercury": "The model identified Mercury's gray rocky surface and crater-rich appearance.",

    "neptune": "The model focused on Neptune's deep blue coloration and planetary structure.",

    "pluto": "The model recognized Pluto's icy appearance and dwarf planet characteristics.",

    "saturn": "The model focused on the ring system and planetary body, which are distinctive features of Saturn.",

    "uranus": "The model detected Uranus' pale blue color and smooth planetary appearance.",

    "venus": "The model identified Venus' bright cloudy atmosphere and characteristic coloration."
}

# Load Model
model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    12
)

model.load_state_dict(
    torch.load("models/best_model.pth")
)

model.eval()

# Image Path
image_path = r"C:\Users\Manasa\OneDrive\Desktop\SpaceObjectClassification\dataset\spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\test\saturn\saturn (184).jpg"


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Image
image = Image.open(image_path).convert("RGB")

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Prediction
with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(
        probabilities,
        1
    )

predicted_class_name = class_names[predicted.item()]


# Results
print("\n========== SPACE OBJECT CLASSIFICATION ==========")

print("\nPrediction:")
print(predicted_class_name.upper())

print(
    f"\nConfidence: {confidence.item()*100:.2f}%"
)

print("\nExplanation:")
print(explanations[predicted_class_name])

print("\n================================================")