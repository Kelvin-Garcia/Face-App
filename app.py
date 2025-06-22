from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import os

# Cargar el modelo entrenado
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read("modelo_lbph.xml")

# Cargar nombres
nombres = np.load("nombres.npy", allow_pickle=True).tolist()

# Inicializar la app Flask
app = Flask(__name__)
CORS(app)

# Ruta para reconocer desde imagen
@app.route('/reconocer_imagen', methods=['POST'])
def reconocer_imagen():
    file = request.files['imagen']
    npimg = np.frombuffer(file.read(), np.uint8)
    imagen = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

    resultados = []
    for (x, y, w, h) in caras:
        rostro = gris[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (200, 200))
        id_, conf = reconocedor.predict(rostro)
        nombre = nombres[id_] if conf < 90 else "Desconocido"
        resultados.append({"nombre": nombre, "confianza": float(conf)})

    return jsonify(resultados)

# Generador de video para transmisión en tiempo real
def generar_video():
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = face_cascade.detectMultiScale(gris, 1.3, 5)
        for (x, y, w, h) in caras:
            rostro = gris[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (200, 200))
            id_, conf = reconocedor.predict(rostro)
            nombre = nombres[id_] if conf < 90 else "Desconocido"
            cv2.putText(frame, f'{nombre} ({conf:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

@app.route('/video')
def video():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "API de reconocimiento facial operativa."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
