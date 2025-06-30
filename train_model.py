import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# Paths and parameters
IMAGE_DIR = "dataset/Animals"
MODEL_PATH = "app/model/cnn_animal_model.h5"
ENCODER_PATH = "app/model/label_encoder.pkl"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10


def create_data_generators(image_dir, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    train_gen = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_gen, val_gen


def save_label_encoder(class_indices, encoder_path):
    encoder = LabelEncoder()
    encoder.classes_ = np.array(list(class_indices.keys()))
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(encoder, encoder_path)
    print(f"[INFO] Label encoder saved at {encoder_path}")
    return encoder


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model():
    print("[INFO] Loading data...")
    train_gen, val_gen = create_data_generators(IMAGE_DIR, IMG_SIZE, BATCH_SIZE)

    print("[INFO] Saving label encoder...")
    encoder = save_label_encoder(train_gen.class_indices, ENCODER_PATH)

    print("[INFO] Building CNN model...")
    model = build_cnn_model((*IMG_SIZE, 3), len(train_gen.class_indices))

    print("[INFO] Training model...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    print("[INFO] Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] CNN model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
