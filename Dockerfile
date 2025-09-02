# Imagen base con Python 3.10
FROM python:3.10-slim

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/tmp

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de la app
WORKDIR /app

# Copiar requirements primero (para cache)
COPY requirements.txt .

# Crear entorno virtual dentro del contenedor
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar pip y dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto para Railway
EXPOSE 8080

# Comando de inicio
CMD ["uvicorn", "api_fingerprint:app", "--host", "0.0.0.0", "--port", "8080"]
