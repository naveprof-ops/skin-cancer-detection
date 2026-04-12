import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import EfficientNetB0

images = glob('dataset/*/*.jpg')
images = [path.replace('\\', '/') for path in images]

df = pd.DataFrame({'filepath': images})
df['label'] = df['filepath'].str.split('/', expand=True)[1]
df['label_bin'] = np.where(df['label'] == 'malignant', 1, 0)

X_train, X_val, Y_train, Y_val = train_test_split(
    df['filepath'], df['label_bin'],
    test_size=0.2,
    random_state=42
)

def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = train_ds.map(preprocess_image).shuffle(1000).batch(32)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_ds = val_ds.map(preprocess_image).batch(32)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_train),
    y=Y_train
)

class_weights = dict(enumerate(class_weights))

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

model.save("model.h5")

print("Training completed!")