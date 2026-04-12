import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Title
st.title("Skin Cancer Detection App")

# Load model
model = keras.models.load_model("model.h5")

# Upload image
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.error("Malignant (Cancer)")
    else:
        st.success("Benign (Safe)")