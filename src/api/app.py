from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os

# Déterminer les chemins absolus des dossiers
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
TEMPLATE_DIR = os.path.join(PROJECT_DIR, "templates")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

# Charger le modèle entraîné
MODEL_PATH = "models/cnn_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas. Assurez-vous d'avoir bien entraîné et sauvegardé le modèle.")

print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)

# Définition des classes (exemple, à adapter)
CLASSES = {i: f"Classe {i}" for i in range(43)}

# Définition de l'application Flask
app = Flask(__name__)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


# Dossier d'upload des images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route principale (Interface web)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Vérifier si un fichier a été envoyé
        if "file" not in request.files:
            return render_template("index.html", error="Aucun fichier envoyé.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Nom de fichier invalide.")

        # Sauvegarde de l'image
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Chargement et prétraitement de l'image
        image = Image.open(img_path)
        image = image.resize((32, 32))
        image = np.array(image) / 255.0  # Normalisation
        image = np.expand_dims(image, axis=0)  # Ajouter la dimension batch

        # Prédiction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return render_template("index.html", img_path=img_path, 
                               predicted_class=CLASSES[predicted_class], confidence=confidence)
    
    return render_template("index.html")

# Route pour faire une prédiction via API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nom de fichier invalide"}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((32, 32))
        image = np.array(image) / 255.0  # Normalisation
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return jsonify({"predicted_class": int(predicted_class), "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
