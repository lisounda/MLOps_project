import os
import mlflow
import mlflow.tensorflow
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Définition des paramètres
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_PATH = "data/train"
MODEL_PATH = "models/cnn_model"

# Charger et prétraiter les images
def load_data(train_path):
    images, labels = [], []
    for class_id in range(43):  # GTSRB a 43 classes
        class_path = os.path.join(train_path, str(class_id))
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(class_id)

    images = np.array(images) / 255.0  # Normalisation
    labels = to_categorical(np.array(labels), num_classes=43)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
def load_test_data(test_path):
    test_images = []
    test_filenames = []

    for img_name in os.listdir(test_path):
        img_path = os.path.join(test_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        test_images.append(img)
        test_filenames.append(img_name)  # Pour garder la correspondance avec les prédictions

    test_images = np.array(test_images) / 255.0  # Normalisation
    return test_images, test_filenames


# Définition du modèle CNN
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(43, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Fonction principale d'entraînement
def train():
    X_train, X_val, y_train, y_val = load_data(TRAIN_PATH)

    model = build_model()

    # Activer MLflow
    mlflow.set_experiment("Traffic_Sign_Recognition")
    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        # Entraînement du modèle
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

        # Enregistrement des métriques
        val_loss, val_acc = model.evaluate(X_val, y_val)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)

        # Sauvegarde du modèle avec MLflow
        mlflow.tensorflow.log_model(model, "traffic_sign_model")

        print(f"Modèle sauvegardé avec une accuracy de {val_acc:.4f}")

if __name__ == "__main__":
    train()
