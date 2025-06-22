# test_knn.py
import os
import cv2
import numpy as np
import joblib
from deepface import DeepFace

# Config
MODELO = "Facenet"
DETECTOR = "mtcnn"
IMAGEN_PATH = "fotos_normalizadas/BALTODANO KARLITA/Baltodano Karlita.jpg"
UMBRAL_CONFIANZA = 50  # porcentaje mínimo para considerar conocido

modelo = joblib.load("modelo_knn.pkl")
nombres = np.load("nombres_knn.npy", allow_pickle=True)

img_bgr = cv2.imread(IMAGEN_PATH)
if img_bgr is None:
    print(f"[ERROR] No se pudo cargar: {IMAGEN_PATH}")
    exit(1)

cv2.imwrite("temp_test.jpg", img_bgr)

try:
    emb = DeepFace.represent(
        img_path="temp_test.jpg",
        model_name=MODELO,
        detector_backend=DETECTOR,
        enforce_detection=False
    )

    os.remove("temp_test.jpg")

    if not emb or not isinstance(emb[0], dict):
        raise ValueError("No se detectó rostro válido.")

    emb_vec = emb[0]["embedding"]
    pred = modelo.predict([emb_vec])[0]
    dist, indices = modelo.kneighbors([emb_vec], n_neighbors=1, return_distance=True)
    confianza = round(100 - dist[0][0], 2)

    nombre = nombres[pred] if confianza >= UMBRAL_CONFIANZA else "Desconocido"
    print(f"Predicción: {nombre}")
    print(f"Confianza: {confianza}%")

except Exception as e:
    print(f"[ERROR] {e}")





