import requests

# URL de ton API (remplace avec ton IP publique)
url = "http://10.191.173.148:8080/predict"

# Chemin de l'image à envoyer (mets une image dans ton projet)
image_path = "data/Test/00000.png"

# Envoi de la requête POST avec le fichier image
with open(image_path, "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files)

# Affichage de la réponse
print(response.json())
