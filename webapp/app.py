from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# ==========================
# CLASS NAMES
# ==========================

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

# ==========================
# EXPLANATIONS
# ==========================

explanations = {
    "asteroid": "The model identified irregular rocky structures and asteroid-like surface patterns.",

    "black_hole": "The model focused on the dark central region and surrounding accretion features typical of black holes.",

    "earth": "The model detected Earth's characteristic blue oceans, clouds, and continental structures.",

    "galaxy": "The model focused on the bright galactic core and spiral stellar structures.",

    "jupiter": "The model recognized Jupiter's large planetary body and atmospheric band patterns.",

    "mars": "The model detected the reddish surface coloration typical of Mars.",

    "mercury": "The model identified Mercury's gray rocky surface and crater-rich appearance.",

    "neptune": "The model focused on Neptune's deep blue coloration and planetary structure.",

    "pluto": "The model recognized Pluto's icy appearance and dwarf planet characteristics.",

    "saturn": "The model focused on the ring system and planetary body, which are distinctive features of Saturn.",

    "uranus": "The model detected Uranus' pale blue color and smooth planetary appearance.",

    "venus": "The model identified Venus' bright cloudy atmosphere and characteristic coloration."
}

# ==========================
# LOAD MODEL
# ==========================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    12
)

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "best_model.pth"
)

model.load_state_dict(
    torch.load(model_path, map_location=torch.device("cpu"))
)

model.eval()

# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    html_path = os.path.join(
        current_dir,
        "index.html"
    )

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ==========================
# PREDICTION API
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    image = Image.open(file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        print("\n===== PROBABILITIES =====")

        for i, prob in enumerate(probabilities[0]):
            print(
                class_names[i],
                ":",
                round(prob.item() * 100, 2),
                "%"
            )

        confidence, predicted = torch.max(
            probabilities,
            1
        )
    predicted_class = class_names[
        predicted.item()
    ]

    # ==========================
    # GRAD-CAM GENERATION
    # ==========================

    file.seek(0)

    image_for_cam = Image.open(file).convert("RGB")

    rgb_img = np.array(
        image_for_cam.resize((224, 224))
    ) / 255.0

    input_tensor = transform(
        image_for_cam
    ).unsqueeze(0)

    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted.item()
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    gradcam_dir = os.path.join(
        current_dir,
        "static",
        "gradcam"
    )

    os.makedirs(
        gradcam_dir,
        exist_ok=True
    )

    original_path = os.path.join(
        gradcam_dir,
        "original.jpg"
    )

    heatmap_path = os.path.join(
        gradcam_dir,
        "heatmap.jpg"
    )

    cv2.imwrite(
        original_path,
        cv2.cvtColor(
            (rgb_img * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return jsonify({
        "prediction": predicted_class.upper(),

        "confidence": round(
            confidence.item() * 100,
            2
        ),

        "explanation": explanations[
            predicted_class
        ],

        "original_image":
            "/static/gradcam/original.jpg",

        "heatmap_image":
            "/static/gradcam/heatmap.jpg"
    })

# ==========================
# RUN APP
# ==========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )