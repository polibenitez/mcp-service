import logging
from rag_pipeline import RAGPipeline, process_local_json_file
import requests
import os
# Importar configuración
from src.config import load_config
import json
import random

# Cargar configuración
print("cargamos config")
config = load_config()
print("cargamos config terminada")
proxies = { 
              "http"  : "http://madrovi:Data123$@ps-sev-sys.cec.eu.int:8012", 
              "https" :"http://madrovi:Data123$@ps-sev-sys.cec.eu.int:8012"
            }
def configure_proxy():
    scheme = 'http'
    user = config["USER_PROXY"]
    password = config["PASS_PROXY"]
    host = 'ps-sev-sys.cec.eu.int'
    port = 8012

    # # Construct the proxy URL
    proxy_url = f"{scheme}://{user}:{password}@{host}:{port}"
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url


def main():
    # Inicializar el pipeline RAG
    print("main")
    pipeline = RAGPipeline()
    
    # Recuperar datos de la API
    #response = requests.get('https://knowledge4policy.ec.europa.eu/api-gpt-jrc_en', proxies=proxies)
    #if response.status_code != 200:
     #   logging.error(f'Error al obtener datos de la API: {response.status_code}')
    #    return
    #publications = response.json()


    # Ruta al archivo JSON
    archivo_json = 'data/publications.json'

    # Abrir y leer el archivo JSON
    with open(archivo_json, 'r', encoding='utf-8') as archivo:
        publications = json.load(archivo)


    # Seleccionar una muestra aleatoria de 5 publicaciones
    muestra = random.sample(publications, 5)

    # Guardar la muestra en un nuevo archivo JSON
    with open('muestra_publicaciones.json', 'w', encoding='utf-8') as archivo_salida:
        json.dump(muestra, archivo_salida, ensure_ascii=False, indent=4)

    # Extraer contenido relevante
    processed_documents = []
    for pub in publications:
        doc = pipeline.extract_relevant_content(pub)
        processed_documents.append(doc)
    
    # Indexar documentos en Qdrant
    pipeline.index_documents(processed_documents)
    
    # Realizar algunas consultas de ejemplo
    ejemplos_consultas = [
        "¿Qué es el Circularity Gap Report?",
        "¿Cuál es el objetivo principal del informe?",
        "¿Quién publica el Circularity Gap Report?",
        "¿Con qué frecuencia se publica este informe?"
    ]
    
    for query in ejemplos_consultas:
        print(f"\n\n{'='*50}")
        print(f"CONSULTA: {query}")
        print(f"{'='*50}")
        
        # Obtener y mostrar respuesta
        response = pipeline.run_rag_query_with_llm(query)
        print("\nRESPUESTA:")
        print(response)

if __name__ == "__main__":
    configure_proxy()
    main()