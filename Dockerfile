# Utiliser une image Python comme base
FROM python:3.12

# Installer les dépendances système pour OpenCV et autres bibliothèques
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le dossier templates pour les fichiers HTML
COPY templates/ templates/

# Copier le reste du code
COPY . .

# Exposer le port utilisé par l’application
EXPOSE 8080

# Commande pour lancer l’application
CMD ["python", "src/api/app.py"]
