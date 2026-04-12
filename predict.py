import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("model.h5")

def predict_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    st.write("Prediction value:", pred)

    if pred > 0.4:
        print("Malignant (Cancer)")
    else:
        print("Benign (Safe)")

predict_image("test.jpg")