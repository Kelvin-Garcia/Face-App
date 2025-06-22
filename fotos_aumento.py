# entrenamiento_knn.py
import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Config
RUTA_FOTOS = "fotos_aumentadas"
MODELO = "Facenet"
DETECTOR = "mtcnn"
EMBEDDINGS = []
ETIQUETAS = []

print("[INFO] Extrayendo embeddings con DeepFace...")

for persona in sorted(os.listdir(RUTA_FOTOS)):
    ruta_persona = os.path.join(RUTA_FOTOS, persona)
    if not os.path.isdir(ruta_persona):
        continue

    for archivo in os.listdir(ruta_persona):
        ruta_imagen = os.path.join(ruta_persona, archivo)
        try:
            img_bgr = cv2.imread(ruta_imagen)
            if img_bgr is None:
                print(f"[ERROR] No se pudo leer: {ruta_imagen}")
                continue

            temp_path = "temp_knn.jpg"
            cv2.imwrite(temp_path, img_bgr)

            emb = DeepFace.represent(
                img_path=temp_path,
                model_name=MODELO,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            os.remove(temp_path)

            if isinstance(emb, list) and "embedding" in emb[0]:
                EMBEDDINGS.append(emb[0]["embedding"])
                ETIQUETAS.append(persona)
                print(f"[OK] Embedding extraído: {persona}/{archivo}")

        except Exception as e:
            print(f"[ERROR] Fallo en {archivo}: {e}")

# Entrenamiento
le = LabelEncoder()
y = le.fit_transform(ETIQUETAS)
X = np.array(EMBEDDINGS)

print("[INFO] Entrenando KNN...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

joblib.dump(model, "modelo_knn.pkl")
np.save("nombres_knn.npy", le.classes_)
print(f"[OK] Entrenamiento KNN completado. Total personas: {len(le.classes_)}")
