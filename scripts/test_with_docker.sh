#!/bin/bash
# Script para probar el servicio MCP-RAG en entorno Docker

# Colores para salida
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Prueba de MCP-RAG con Docker =====${NC}"

# Verificar que Docker y Docker Compose están instalados
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker y/o Docker Compose no están instalados${NC}"
    exit 1
fi

# Verificar que existe el archivo .env
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: Archivo .env no encontrado${NC}"
    echo "Por favor, crea un archivo .env con las variables necesarias (ver .env.example)"
    exit 1
fi

# Verificar que existe el archivo de datos
if [ ! -f "paste.txt" ]; then
    echo -e "${RED}Error: Archivo de datos paste.txt no encontrado${NC}"
    exit 1
fi

# Crear directorio de datos si no existe
mkdir -p data

# Copiar archivo de datos al directorio que se montará en Docker
echo "Copiando archivo de datos a directorio data/"
cp paste.txt data/

# Iniciar servicios con Docker Compose
echo -e "\n${YELLOW}[1] Iniciando servicios con Docker Compose...${NC}"
docker-compose up -d

# Verificar si los servicios están en ejecución
echo -e "\n${YELLOW}[2] Verificando estado de los servicios...${NC}"
if docker-compose ps | grep -q "mcp-rag.*Up"; then
    echo -e "${GREEN}✓ Servicio mcp-rag en ejecución${NC}"
else
    echo -e "${RED}✗ Error: Servicio mcp-rag no está en ejecución${NC}"
    docker-compose logs mcp-rag
    docker-compose down
    exit 1
fi

if docker-compose ps | grep -q "qdrant.*Up"; then
    echo -e "${GREEN}✓ Servicio Qdrant en ejecución${NC}"
else
    echo -e "${RED}✗ Error: Servicio Qdrant no está en ejecución${NC}"
    docker-compose logs qdrant
    docker-compose down
    exit 1
fi

# Esperar a que los servicios estén completamente iniciados
echo -e "\n${YELLOW}[3] Esperando a que los servicios estén listos...${NC}"
echo "Esperando 10 segundos..."
sleep 10

# Probar la API de salud
echo -e "\n${YELLOW}[4] Verificando estado de la API...${NC}"
HEALTH_CHECK=$(curl -s http://localhost:8000/health)

if [[ $HEALTH_CHECK == *"ok"* ]]; then
    echo -e "${GREEN}✓ API funcionando correctamente${NC}"
else
    echo -e "${RED}✗ Error: API no responde correctamente${NC}"
    echo "Respuesta: $HEALTH_CHECK"
    docker-compose logs mcp-rag
    docker-compose down
    exit 1
fi

# Indexar archivo
echo -e "\n${YELLOW}[5] Indexando archivo de datos...${NC}"
INDEX_RESPONSE=$(curl -s -X POST "http://localhost:8000/index/file" \
    -H "Content-Type: application/json" \
    -d '{"file_path": "/app/data/paste.txt"}')

if [[ $INDEX_RESPONSE == *"success"* ]]; then
    echo -e "${GREEN}✓ Archivo indexado correctamente${NC}"
else
    echo -e "${RED}✗ Error al indexar archivo${NC}"
    echo "Respuesta: $INDEX_RESPONSE"
    docker-compose logs mcp-rag
fi

# Esperar a que termine la indexación
echo "Esperando 5 segundos para que finalice la indexación..."
sleep 5

# Realizar una consulta
echo -e "\n${YELLOW}[6] Realizando consulta de prueba...${NC}"
QUERY_RESPONSE=$(curl -s -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "¿Qué es el Circularity Gap Report?", "limit": 3}')

if [[ $QUERY_RESPONSE == *"answer"* ]]; then
    echo -e "${GREEN}✓ Consulta procesada correctamente${NC}"
    # Extraer la respuesta (esta es una forma simplificada)
    echo -e "\nRespuesta obtenida del sistema RAG:"
    echo $QUERY_RESPONSE | grep -o '"answer":"[^"]*"' | sed 's/"answer":"//;s/"$//' | fold -w 100
else
    echo -e "${RED}✗ Error al procesar consulta${NC}"
    echo "Respuesta: $QUERY_RESPONSE"
    docker-compose logs mcp-rag
fi

# Preguntar al usuario si desea detener los servicios
echo -e "\n${YELLOW}[7] Prueba completada${NC}"
read -p "¿Deseas detener los servicios de Docker? (s/n): " STOP_SERVICES

if [[ $STOP_SERVICES == "s" || $STOP_SERVICES == "S" ]]; then
    echo "Deteniendo servicios..."
    docker-compose down
    echo -e "${GREEN}Servicios detenidos correctamente${NC}"
else
    echo -e "${YELLOW}Los servicios siguen en ejecución. Puedes detenerlos manualmente con 'docker-compose down'${NC}"
fi

echo -e "\n${GREEN}===== Prueba finalizada =====${NC}"