# app.py (para despliegue en Render)
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
from deepface import DeepFace
import os

# Config
MODELO = "Facenet"
DETECTOR = "mtcnn"
UMBRAL_CONFIANZA = 50

# Cargar modelo y etiquetas
modelo = joblib.load("modelo_knn.pkl")
nombres = np.load("nombres_knn.npy", allow_pickle=True)

app = Flask(__name__)
CORS(app)

@app.route("/reconocer_imagen", methods=["POST"])
def reconocer():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen."}), 400

    archivo = request.files['imagen']
    npimg = np.frombuffer(archivo.read(), np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "No se pudo decodificar la imagen."}), 400

    # Guardar temporal para usar en DeepFace
    temp_path = "temp_api.jpg"
    cv2.imwrite(temp_path, img_bgr)

    try:
        emb = DeepFace.represent(
            img_path=temp_path,
            model_name=MODELO,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

        os.remove(temp_path)

        if not emb or not isinstance(emb[0], dict):
            raise ValueError("No se detectó rostro válido.")

        emb_vec = emb[0]["embedding"]
        pred = modelo.predict([emb_vec])[0]

        dist, indices = modelo.kneighbors([emb_vec], n_neighbors=1, return_distance=True)
        confianza = round(100 - dist[0][0], 2)
        nombre = nombres[pred] if confianza >= UMBRAL_CONFIANZA else "Desconocido"

        return jsonify({
            "nombre": nombre,
            "confianza": confianza
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "API de reconocimiento facial KNN operativa"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
