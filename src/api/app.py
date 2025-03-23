from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import random

# Déterminer les chemins absolus
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
TEMPLATE_DIR = os.path.join(PROJECT_DIR, "templates")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")

# Chargement du modèle
MODEL_PATH = "models/cnn_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas.")

print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)

# Dictionnaire des classes
panneaux_signification = {
    "0": "Limite de vitesse (20 km/h)", "1": "Limite de vitesse (30 km/h)", "2": "Limite de vitesse (50 km/h)",
    "3": "Limite de vitesse (60 km/h)", "4": "Limite de vitesse (70 km/h)", "5": "Limite de vitesse (80 km/h)",
    "6": "Fin de la limite de vitesse (80 km/h)", "7": "Limite de vitesse (100 km/h)", "8": "Limite de vitesse (120 km/h)",
    "9": "Dépassement interdit", "10": "Dépassement interdit pour les poids lourds", "11": "Intersection prioritaire",
    "12": "Route prioritaire", "13": "Céder le passage", "14": "Stop", "15": "Circulation interdite",
    "16": "Interdiction aux poids lourds", "17": "Accès interdit", "18": "Danger général",
    "19": "Virage dangereux à gauche", "20": "Virage dangereux à droite", "21": "Virages successifs",
    "22": "Dos-d’âne ou ralentisseur", "23": "Chaussée glissante", "24": "Réduction de voie",
    "25": "Attention Travaux", "26": "Feux tricolores", "27": "Passage piéton ou traversée fréquente de piétons",
    "28": "Écoliers - Attention aux enfants", "29": "Débouchés de cyclistes", "30": "Risque de verglas ou de neige",
    "31": "Passage d’animaux sauvages", "32": "Fin de toutes les interdictions", "33": "Obligation d'aller à droite",
    "34": "Obligation d'aller à gauche", "35": "Obligation d'aller tout droit", "36": "Obligation d'aller tout droit ou à droite",
    "37": "Obligation d'aller tout droit ou à gauche", "38": "Obligation de contourner par la droite",
    "39": "Obligation de contourner par la gauche", "40": "Sens giratoire obligatoire", "41": "Fin d'interdiction de dépassement",
    "42": "Fin d'interdiction de dépassement pour les poids lourds"
}

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
TEMP_FOLDER = os.path.join(STATIC_DIR, "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="Aucun fichier envoyé.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Nom de fichier invalide.")

        # Sauvegarder l’image uploadée
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Prétraitement
        image = Image.open(img_path).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prédiction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        description = panneaux_signification.get(str(predicted_class), "Panneau inconnu")

        # Sélection aléatoire d’une image du dataset
        train_class_dir = os.path.join("data", "train", str(predicted_class))
        panneau_img_web = None
        if os.path.exists(train_class_dir):
            class_images = os.listdir(train_class_dir)
            if class_images:
                random_image = random.choice(class_images)
                panneau_img_src = os.path.join(train_class_dir, random_image)
                temp_display_path = os.path.join(TEMP_FOLDER, f"class_{predicted_class}.png")
                cv2.imwrite(temp_display_path, cv2.imread(panneau_img_src))
                panneau_img_web = f"temp/class_{predicted_class}.png"

        return render_template("index.html",
                               img_path=f"uploads/{file.filename}",
                               predicted_class=f"Classe {predicted_class}",
                               confidence=round(confidence, 4),
                               description=description,
                               panneau_img_path=panneau_img_web)

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nom de fichier invalide"}), 400

        image = Image.open(io.BytesIO(file.read())).resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        description = panneaux_signification.get(str(predicted_class), "Panneau inconnu")

        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "description": description
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
