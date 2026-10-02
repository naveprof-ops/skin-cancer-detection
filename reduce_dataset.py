import os
import shutil
import random

source_dir = "dataset"

target_dir = "small_dataset"

limit = 200   

os.makedirs(target_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)

        selected = images[:limit]

        new_class_path = os.path.join(target_dir, class_name)
        os.makedirs(new_class_path, exist_ok=True)

        for img in selected:
            src = os.path.join(class_path, img)
            dst = os.path.join(new_class_path, img)
            shutil.copy(src, dst)

print("✅ Small dataset created successfully!")