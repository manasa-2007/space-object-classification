import os
import random
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = r"spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\training"

classes = sorted(os.listdir(dataset_path))

plt.figure(figsize=(15,10))

for i, cls in enumerate(classes):

    class_folder = os.path.join(dataset_path, cls)

    image_name = random.choice(os.listdir(class_folder))

    image_path = os.path.join(class_folder, image_name)

    img = Image.open(image_path)

    plt.subplot(3,4,i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()