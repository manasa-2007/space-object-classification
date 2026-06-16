import os

dataset_path = r"spaceobject_dataset\astro_dataset_maxia\astro_dataset_maxia\training"

classes = sorted(os.listdir(dataset_path))

print("Classes Found:")
print("-" * 30)

for cls in classes:
    class_path = os.path.join(dataset_path, cls)

    if os.path.isdir(class_path):
        image_count = len(os.listdir(class_path))
        print(f"{cls}: {image_count} images")

print("-" * 30)
print(f"Total Classes: {len(classes)}")