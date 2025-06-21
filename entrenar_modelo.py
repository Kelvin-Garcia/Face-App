import os
import cv2
import numpy as np

# Ruta de la carpeta con las fotos organizadas por persona
ruta_fotos = "fotos_normalizadas"

# Inicializamos el reconocedor LBPH
reconocedor = cv2.face.LBPHFaceRecognizer_create()

imagenes = []
etiquetas = []
nombres = []
diccionario_nombres = {}

contador_id = 0

# Recorremos las carpetas de cada persona
for nombre_persona in sorted(os.listdir(ruta_fotos)):
    ruta_persona = os.path.join(ruta_fotos, nombre_persona)

    if not os.path.isdir(ruta_persona):
        continue

    # Asignamos un ID único a cada persona
    if nombre_persona not in diccionario_nombres:
        diccionario_nombres[nombre_persona] = contador_id
        contador_id += 1

    id_persona = diccionario_nombres[nombre_persona]

    # Recorremos las imágenes de esa persona
    for archivo in os.listdir(ruta_persona):
        ruta_imagen = os.path.join(ruta_persona, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

        if imagen is None:
            print(f"[Error] No se pudo leer la imagen: {ruta_imagen}")
            continue

        imagenes.append(imagen)
        etiquetas.append(id_persona)

# Entrenamos el modelo
if len(imagenes) > 0:
    reconocedor.train(imagenes, np.array(etiquetas))
    reconocedor.save("modelo_lbph.xml")

    # Guardamos los nombres en el orden correcto
    lista_nombres = [None] * len(diccionario_nombres)
    for nombre, id_ in diccionario_nombres.items():
        lista_nombres[id_] = nombre

    np.save("nombres.npy", np.array(lista_nombres))

    print(f"[OK] Entrenamiento completado. Se entrenaron {len(set(etiquetas))} personas.")
    print(f"[OK] Modelo guardado como 'modelo_lbph.xml'")
    print(f"[OK] Nombres guardados en 'nombres.npy'")
else:
    print("[Error] No se encontraron imágenes válidas para entrenar.")
