# Usa imagen ligera de Python 3.10
FROM python:3.10-slim

# Instala dependencias del sistema necesarias para OpenCV y DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea y entra al directorio de la app
WORKDIR /app

# Copia e instala dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia el resto del código
COPY . .

# Define el puerto que Railway usará
ENV PORT=8080

# Comando para correr la app con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
