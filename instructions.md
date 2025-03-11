# Instrucciones de instalación y ejecución

## Estructura de directorios

Para usar correctamente este proyecto, asegúrate de tener esta estructura de directorios:

```
mcp_rag_project/
├── src/               # Código fuente
├── tests/             # Pruebas
├── data/              # Archivos de datos
├── setup.py           # Configuración del paquete
└── ...
```

## Preparación del entorno

### 1. Clonar el repositorio y preparar el entorno virtual

```bash
# Clonar el repositorio (si aplicable)
git clone <url-del-repositorio>
cd mcp_rag_project

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar el paquete en modo desarrollo

Para solucionar los problemas de importación, instala el paquete en modo desarrollo:

```bash
pip install -e .
```

Esta instalación hace que el paquete `src` sea accesible como un módulo en cualquier lugar, lo que facilita las importaciones.

### 3. Configurar variables de entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus valores
nano .env
```

## Ejecutar las pruebas

### Pruebas unitarias y de integración

```bash
# Ejecutar todas las pruebas
python -m unittest discover tests

# Ejecutar una prueba específica
python -m unittest tests.test_mcp_rag
```

### Pruebas de API

```bash
# Instalar dependencias para pruebas
pip install httpx pytest

# Ejecutar pruebas de API
python -m unittest tests.test_api
```

### Pruebas con scripts shell

```bash
# Dar permisos de ejecución
chmod +x tests/curl_test_requests.sh
chmod +x tests/test_with_docker.sh

# Ejecutar pruebas con cURL (el servidor debe estar en ejecución)
./tests/curl_test_requests.sh

# Ejecutar pruebas completas con Docker
./tests/test_with_docker.sh
```

## Ejecutar la aplicación

### Ejecución local

```bash
# Iniciar el servidor API
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Ejecución con Docker

```bash
# Construir y ejecutar contenedores
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## Solución de problemas comunes

### Error "Module not found"

Si después de instalar con `pip install -e .` sigues teniendo problemas con las importaciones, verifica:

1. Que todos los directorios tienen un archivo `__init__.py`
2. Que estás ejecutando Python desde el directorio raíz del proyecto
3. Que el entorno virtual está activado

Solución alternativa:

```python
# Al inicio de cada archivo de prueba
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Errores con dependencias

Si encuentras problemas con las dependencias, intenta:

```bash
pip install -r requirements.txt
```

### Ayuda adicional

Si persisten los problemas, comparte el mensaje de error completo y la información sobre tu entorno:

```bash
# Información del entorno
python --version
pip list
```