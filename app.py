# app.py (Render optimizado con precarga)
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import os
from deepface.basemodels import Facenet
from deepface.detectors import FaceDetector
from deepface.commons import functions
from deepface.commons.logger import Logger
Logger.disabled = True  # Silenciar logs internos

# Config
MODELO = "Facenet"
DETECTOR = "mtcnn"
UMBRAL_CONFIANZA = 50

# Precargar modelo y detector
facenet_model = Facenet.loadModel()
face_detector = FaceDetector.build_model(DETECTOR)
target_size = functions.find_target_size(model_name=MODELO)

# Cargar modelo KNN y nombres
modelo = joblib.load("modelo_knn.pkl")
nombres = np.load("nombres_knn.npy", allow_pickle=True)

# Inicializar Flask
app = Flask(__name__)
CORS(app)

@app.route("/reconocer", methods=["POST"])
def reconocer():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen."}), 400

    archivo = request.files['imagen']
    npimg = np.frombuffer(archivo.read(), np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "No se pudo decodificar la imagen."}), 400

    try:
        # Detección de rostro
        faces = functions.extract_faces(
            img=img_bgr,
            target_size=target_size,
            detector_backend=DETECTOR,
            detector_model=face_detector,
            enforce_detection=False
        )

        if not faces or not isinstance(faces[0], dict):
            raise ValueError("No se detectó rostro válido.")

        rostro = faces[0]["face"]
        embedding = functions.represent_face(
            rostro, model_name=MODELO, model=facenet_model, enforce_detection=False
        )

        # Clasificación KNN
        pred = modelo.predict([embedding])[0]
        dist, _ = modelo.kneighbors([embedding], n_neighbors=1, return_distance=True)
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

@app.route("/health")
def health():
    return jsonify({"status": "ok", "modelo": MODELO, "personas": len(nombres)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
