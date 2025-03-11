FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar archivos de requerimientos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY *.py .
COPY data/ ./data/

# Exponer el puerto para FastAPI
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]