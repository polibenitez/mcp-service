#!/bin/bash
# Script para probar la API del servicio MCP-RAG usando curl

# Configuración
API_URL="http://localhost:8000"
TEST_FILE="paste.txt"

# Colores para salida
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "==== Pruebas de integración para API MCP-RAG ===="

# Verificar que el servicio está en ejecución
echo -e "\n[1] Verificando estado del servicio..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")

if [[ $HEALTH_RESPONSE == *"ok"* ]]; then
    echo -e "${GREEN}✓ Servicio funcionando correctamente${NC}"
else
    echo -e "${RED}✗ Error: Servicio no disponible${NC}"
    echo "Respuesta: $HEALTH_RESPONSE"
    exit 1
fi

# Resetear la base de datos para empezar limpio
echo -e "\n[2] Limpiando base de datos..."
RESET_RESPONSE=$(curl -s -X POST "$API_URL/reset")

if [[ $RESET_RESPONSE == *"success"* ]]; then
    echo -e "${GREEN}✓ Base de datos reiniciada correctamente${NC}"
else
    echo -e "${RED}✗ Error al reiniciar base de datos${NC}"
    echo "Respuesta: $RESET_RESPONSE"
fi

# Indexar archivo de ejemplo
echo -e "\n[3] Indexando archivo $TEST_FILE..."
INDEX_RESPONSE=$(curl -s -X POST "$API_URL/index/file" \
    -H "Content-Type: application/json" \
    -d "{\"file_path\": \"$TEST_FILE\"}")

if [[ $INDEX_RESPONSE == *"success"* ]]; then
    echo -e "${GREEN}✓ Archivo indexado correctamente${NC}"
else
    echo -e "${RED}✗ Error al indexar archivo${NC}"
    echo "Respuesta: $INDEX_RESPONSE"
fi

# Esperar un momento para que termine la indexación en segundo plano
echo "Esperando 5 segundos para que finalice la indexación..."
sleep 5

# Realizar una consulta
echo -e "\n[4] Realizando consulta..."
QUERY_RESPONSE=$(curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "¿Qué es el Circularity Gap Report?", "limit": 3}')

if [[ $QUERY_RESPONSE == *"answer"* ]]; then
    echo -e "${GREEN}✓ Consulta procesada correctamente${NC}"
    
    # Extraer y mostrar la respuesta (usando grep y sed simplificados para este ejemplo)
    ANSWER=$(echo $QUERY_RESPONSE | grep -o '"answer":"[^"]*"' | sed 's/"answer":"//;s/"$//')
    echo -e "\nRespuesta: $ANSWER"
else
    echo -e "${RED}✗ Error al procesar consulta${NC}"
    echo "Respuesta: $QUERY_RESPONSE"
fi

# Realizar consulta sin incluir documentos
echo -e "\n[5] Realizando consulta sin incluir documentos..."
QUERY_NO_DOCS=$(curl -s -X POST "$API_URL/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "¿Qué es el Circularity Gap Report?", "include_documents": false}')

if [[ $QUERY_NO_DOCS == *"documents":[]* ]]; then
    echo -e "${GREEN}✓ Consulta sin documentos procesada correctamente${NC}"
else
    echo -e "${RED}✗ Error: La respuesta incluye documentos o tiene otro error${NC}"
    echo "Respuesta: $QUERY_NO_DOCS"
fi

echo -e "\n==== Pruebas completadas ====\n"