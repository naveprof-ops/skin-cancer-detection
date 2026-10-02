import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# 1. LOAD THE MODEL
model_path = "model.h5"

if os.path.exists(model_path):
    # We use compile=False to avoid errors with custom metrics during loading
    model = keras.models.load_model(model_path, compile=False)
    print("SUCCESS: Model loaded successfully.")
else:
    print(f"ERROR: {model_path} not found. Please wait for train.py to finish!")
    exit()

def predict_skin_image(img_path):
    # Check if the image file actually exists
    if not os.path.exists(img_path):
        print(f"ERROR: The image file '{img_path}' was not found in this folder.")
        return

    # 2. PREPROCESS IMAGE (Matches train.py logic)
    # EfficientNetB0 requires specific steps to give a correct answer
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, (224, 224))      # Resize to 224x224
    img = img / 255.0                           
    img = tf.expand_dims(img, axis=0)           

    # 3. RUN PREDICTION
    pred = model.predict(img)[0][0]
    
    print("\n" + "="*40)
    print(f"RESULTS FOR: {img_path}")
    print(f"Raw AI Output: {pred:.4f}")

    # 4. FINAL RESULT LOGIC
    
    if pred > 0.5:
    
        print(f"AI PREDICTION: MALIGNANT (CANCER)")
        print(f"Confidence Level: {pred:.2%}")
    else:
    
        print(f"AI PREDICTION: BENIGN (SAFE)")
        print(f"Confidence Level: {(1-pred):.2%}")
    print("="*40 + "\n")

# RUNNING THE CODE
if __name__ == "__main__":
    # This specifically looks for test.jpg in your project folder
    predict_skin_image("test.jpg")