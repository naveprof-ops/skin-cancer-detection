# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image
# import io

# app = FastAPI()

# # Load model
# model = None

# def get_model():
#     global model
#     if model is None:
#         model = keras.models.load_model("model.h5")
#     return model

# # Prediction function
# def predict_image(image_bytes):
#     model = get_model()   

#     img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)

#     pred = model.predict(img)[0][0]

#     if pred > 0.5:
#         return "Malignant (Cancer)"
#     else:
#         return "Benign (Safe)"

# # API endpoint
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     result = predict_image(image_bytes)
#     return {"prediction": result}