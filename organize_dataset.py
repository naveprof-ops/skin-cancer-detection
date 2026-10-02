import pandas as pd
import os
import shutil

# 1. Load the metadata
metadata_path = "HAM10000_metadata.csv"
if not os.path.exists(metadata_path):
    print(f"ERROR: {metadata_path} not found in current folder!")
    exit()

df = pd.read_csv(metadata_path)

# 2. Define folders
base_dir = "dataset"
categories = ['benign', 'malignant']
malignant_types = ['mel', 'bcc', 'akiec']

for cat in categories:
    os.makedirs(os.path.join(base_dir, cat), exist_ok=True)

# 3. Source Folders (Check if these names match your folders exactly!)
src_folders = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

copy_count = 0
missing_count = 0

print("Starting to copy images...")

for i, row in df.iterrows():
    img_name = row['image_id'] + ".jpg"
    label = 'malignant' if row['dx'] in malignant_types else 'benign'
    
    # Try to find the image in either part 1 or part 2
    found = False
    for folder in src_folders:
        src_path = os.path.join(folder, img_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(base_dir, label, img_name)
            shutil.copy(src_path, dst_path)
            copy_count += 1
            found = True
            break
    
    if not found:
        missing_count += 1

print(f"Done! Successfully copied: {copy_count} images.")
print(f"Images not found: {missing_count}")