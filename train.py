import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight

# 1. ULTIMATE DATA LOADER (Fixes the "0 images found" error)
def load_data():
    base_path = os.path.join(os.getcwd(), 'small_dataset')
    images = []
    
    # Walk through every folder starting from 'dataset'
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Create a clean path that TensorFlow understands
                full_path = os.path.join(root, file).replace('\\', '/')
                images.append(full_path)
    
    if not images:
        print(f"CRITICAL ERROR: No images found in {base_path}!")
        print("Please check your VS Code sidebar to ensure 'dataset' contains images.")
        exit()
        
    df = pd.DataFrame({'filepath': images})
    
    # Identify label based on folder name in path
    df['label'] = df['filepath'].apply(lambda x: 'malignant' if 'malignant' in x.lower() else 'benign')
    df['label_bin'] = np.where(df['label'] == 'malignant', 1, 0)
    
    print(f"Successfully found {len(df)} images.")
    print(f"Malignant cases: {sum(df['label_bin'])}")
    print(f"Benign cases: {len(df) - sum(df['label_bin'])}")
    return df

df = load_data()

# 2. STRATIFIED SPLIT
X_train, X_val, Y_train, Y_val = train_test_split(
    df['filepath'], df['label_bin'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_bin']
)

# 3. AUGMENTATION & PREPROCESSING (Fixes the bias)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img, label

# 4. DATASETS
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = train_ds.map(preprocess_image).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_ds = val_ds.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# 5. MODEL ARCHITECTURE (Transfer Learning)
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-20]: # Keep most weights, train the top 20 layers
    layer.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[keras.metrics.AUC(name='auc'), 'accuracy']
)

# 6. CLASS WEIGHTS
weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weights_dict = dict(enumerate(weights))

# 7. START TRAINING
print("Training started... watching AUC metric for bias correction.")
model.fit(train_ds, validation_data=val_ds, epochs=8, class_weight=class_weights_dict)

model.save("model.h5")
print("Project fixed! model.h5 has been updated.")