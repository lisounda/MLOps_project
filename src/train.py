import os
import mlflow
import mlflow.tensorflow
import numpy as np
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# Définition des paramètres
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 10
TRAIN_PATH = "data/train"
MODEL_DIR = "models"
MODEL_H5_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
MODEL_PKL_PATH = os.path.join(MODEL_DIR, "cnn_model.pkl")

# Vérifier et créer le dossier models/ si besoin
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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

# Définition du modèle CNN
def build_model():
    model = Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(43, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Fonction principale d'entraînement
def train():
    print("📥 Chargement des données...")
    X_train, X_val, y_train, y_val = load_data(TRAIN_PATH)

    print("🔨 Construction du modèle CNN...")
    model = build_model()

    # Activer MLflow
    mlflow.set_experiment("Traffic_Sign_Recognition")
    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)

        print("🚀 Entraînement du modèle...")
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

        # Évaluation
        val_loss, val_acc = model.evaluate(X_val, y_val)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_acc)

        # Sauvegarde du modèle en .h5
        print(f"💾 Sauvegarde du modèle en {MODEL_H5_PATH}...")
        model.save(MODEL_H5_PATH)

        # Sauvegarde du modèle en .pkl pour Flask
        print(f"💾 Sauvegarde du modèle en {MODEL_PKL_PATH}...")
        with open(MODEL_PKL_PATH, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ Modèle sauvegardé avec une accuracy de {val_acc:.4f}")

if __name__ == "__main__":
    train()
