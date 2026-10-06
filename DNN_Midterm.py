import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

def load_image(image_path, label):
    image_path = image_path.decode()

    image = Image.open(image_path)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0

    label = np.int32(label)
    return image, label


def tf_load_image(image_path, label):
    image, label = tf.numpy_function(
        load_image,
        [image_path, label],
        [tf.float32, tf.int32]
    )

    image.set_shape((96, 96, 3))
    label.set_shape(())

    return image, label

def make_dataset(frame, batch=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            frame["image_path"].values,
            frame["label"].values.astype(np.int32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(len(frame))

    dataset = dataset.map(
        tf_load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(batch).prefetch(tf.data.AUTOTUNE)

base_path = "..\\..\\histopathologic-cancer-detection\\"
df = pd.read_csv(base_path + "train_labels.csv")

train_path = base_path + "train\\"
test_path = base_path + "test\\"

df["image_path"] = train_path + df["id"] + ".tif"

train_df, validation_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = make_dataset(train_df, shuffle=True)
validation_dataset = make_dataset(validation_df)

'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96, 96, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

model.save("model.keras")
'''

model = tf.keras.models.load_model("model.keras")

test_labels = pd.read_csv(base_path + "sample_submission.csv")
test_labels["image_path"] = test_path + test_labels["id"] + ".tif"
test_labels["label"] = 0

test_dataset = make_dataset(test_labels)

test_labels["label"] = model.predict(test_dataset).ravel()
test_labels[["id", "label"]].to_csv("submission.csv", index=False)