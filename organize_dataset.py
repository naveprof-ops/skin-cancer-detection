import pandas as pd
import os
import shutil

# Load CSV
df = pd.read_csv("HAM10000_metadata.csv")

# Create folders
os.makedirs("dataset/benign", exist_ok=True)
os.makedirs("dataset/malignant", exist_ok=True)

malignant = ['mel', 'bcc', 'akiec']

count = 0

for i, row in df.iterrows():
    img_name = row['image_id'] + ".jpg"
    label = row['dx']

    # Correct paths
    src1 = os.path.join("HAM10000_images_part_1", img_name)
    src2 = os.path.join("HAM10000_images_part_2", img_name)

    if os.path.exists(src1):
        src = src1
    elif os.path.exists(src2):
        src = src2
    else:
        continue   # skip if not found

    if label in malignant:
        dst = os.path.join("dataset/malignant", img_name)
    else:
        dst = os.path.join("dataset/benign", img_name)

    shutil.copy(src, dst)
    count += 1

print("Total images copied:", count)