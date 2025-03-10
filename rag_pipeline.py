import json
import requests
import os
from typing import List, Dict, Any
import numpy as np

# Para vectorización y conexión con OpenAI
from openai import OpenAI

# Para la base de datos vectorial Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Importar configuración
from config import load_config

# Cargar configuración
config = load_config()

# Acceder a la configuración
OPENAI_API_KEY = config["OPENAI_API_KEY"]
OPENAI_API_BASE = config["OPENAI_API_BASE"]
API_ENDPOINT = config["API_ENDPOINT"]
QDRANT_HOST = config["QDRANT_HOST"]
QDRANT_PORT = config["QDRANT_PORT"]
COLLECTION_NAME = config["COLLECTION_NAME"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
LLM_MODEL = config["LLM_MODEL"]
LLM_TEMPERATURE = config["LLM_TEMPERATURE"]
VECTOR_SIZE = config["VECTOR_SIZE"]

# Configurar cliente OpenAI con base URL opcional
openai_client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_API_BASE:
    openai_client_kwargs["base_url"] = OPENAI_API_BASE

# Inicializar clientes
openai_client = OpenAI(**openai_client_kwargs)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

class RAGPipeline:
    def __init__(self):
        self._setup_qdrant()
    
    def _setup_qdrant(self):
        """Configurar la colección en Qdrant si no existe."""
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Creando colección {COLLECTION_NAME}...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, 
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000  # Umbral para indexación
                )
            )
            print(f"Colección {COLLECTION_NAME} creada correctamente")
    
    def fetch_publications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener publicaciones desde la API."""
        print("Descargando publicaciones desde la API...")
        response = requests.get(f"{API_ENDPOINT}?limit={limit}")
        if response.status_code == 200:
            publications = response.json()
            print(f"Se descargaron {len(publications)} publicaciones")
            return publications
        else:
            raise Exception(f"Error al obtener publicaciones: {response.status_code}")
    
    def extract_relevant_content(self, publication: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraer el contenido relevante de una publicación para vectorización.
        Adaptado para el formato específico de los datos compartidos.
        """
        # Inicializar objeto para almacenar contenido relevante
        relevant_content = {
            "id": None,
            "title": "",
            "summary": "",
            "body": "",
            "metadata": {}
        }
        
        # Extraer ID
        if "uuid" in publication and len(publication["uuid"]) > 0:
            relevant_content["id"] = publication["uuid"][0]["value"]
        elif "nid" in publication and len(publication["nid"]) > 0:
            relevant_content["id"] = str(publication["nid"][0]["value"])
        
        # Extraer título
        if "title" in publication and len(publication["title"]) > 0:
            relevant_content["title"] = publication["title"][0]["value"]
        
        # Extraer cuerpo del texto
        if "body" in publication and len(publication["body"]) > 0:
            body_entry = publication["body"][0]
            relevant_content["body"] = body_entry.get("processed", "")
            
            # Extraer resumen si existe
            if "summary" in body_entry and body_entry["summary"]:
                relevant_content["summary"] = body_entry["summary"]
        
        # Metadatos adicionales que pueden ser útiles
        metadata = {}
        
        # Fecha de creación
        if "created" in publication and len(publication["created"]) > 0:
            metadata["created_date"] = publication["created"][0]["value"]
        
        # Fecha de última actualización
        if "changed" in publication and len(publication["changed"]) > 0:
            metadata["last_updated"] = publication["changed"][0]["value"]
        
        # Cobertura geográfica
        if "field_geographic_coverage" in publication and len(publication["field_geographic_coverage"]) > 0:
            geographic_coverage = []
            for coverage in publication["field_geographic_coverage"]:
                if "target_id" in coverage:
                    geographic_coverage.append(str(coverage["target_id"]))
            metadata["geographic_coverage"] = geographic_coverage
        
        # Organizaciones relacionadas
        if "field_related_organisations" in publication and len(publication["field_related_organisations"]) > 0:
            related_orgs = []
            for org in publication["field_related_organisations"]:
                if "url" in org:
                    related_orgs.append(org["url"])
            metadata["related_organisations"] = related_orgs
        
        # Enlaces externos
        if "field_legacy_link" in publication and len(publication["field_legacy_link"]) > 0:
            external_links = []
            for link in publication["field_legacy_link"]:
                if "uri" in link:
                    external_links.append(link["uri"])
            metadata["external_links"] = external_links
        
        relevant_content["metadata"] = metadata
        
        return relevant_content
    
    def create_embedding(self, text: str) -> List[float]:
        """Crear un embedding para el texto usando OpenAI."""
        if not text.strip():
            return np.zeros(VECTOR_SIZE).tolist()  # Vector de ceros para texto vacío
            
        try:
            response = openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return np.zeros(VECTOR_SIZE).tolist()  # Vector de ceros en caso de error
    
    def create_document_embedding(self, document: Dict[str, Any]) -> List[float]:
        """Crear embedding para un documento completo concatenando campos importantes."""
        # Concatenar título, resumen y cuerpo para crear un texto representativo
        combined_text = document["title"]
        
        if document["summary"]:
            combined_text += " " + document["summary"]
        
        if document["body"]:
            combined_text += " " + document["body"]
        
        return self.create_embedding(combined_text)
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Indexar documentos en Qdrant."""
        print(f"Indexando {len(documents)} documentos en Qdrant...")
        
        points = []
        
        for doc in documents:
            # Crear embedding para el documento
            embedding = self.create_document_embedding(doc)
            
            # Preparar punto para Qdrant
            point = models.PointStruct(
                id=doc["id"] if isinstance(doc["id"], str) else str(doc["id"]),
                vector=embedding,
                payload={
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "body": doc["body"],
                    "metadata": doc["metadata"]
                }
            )
            
            points.append(point)
        
        # Realizar la inserción por lotes
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Indexación completada: {len(points)} documentos añadidos a Qdrant")
    
    def rag_query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Realizar una consulta RAG:
        1. Vectorizar la consulta
        2. Buscar documentos similares en Qdrant
        3. Devolver los documentos relevantes
        """
        # Crear embedding para la consulta
        query_embedding = self.create_embedding(query)
        
        # Buscar documentos similares en Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Preparar respuesta
        results = []
        for scored_point in search_result:
            # Extraer información del punto
            doc = {
                "id": scored_point.id,
                "score": scored_point.score,  # Puntuación de similitud
                "title": scored_point.payload.get("title", ""),
                "summary": scored_point.payload.get("summary", ""),
                "body": scored_point.payload.get("body", ""),
                "metadata": scored_point.payload.get("metadata", {})
            }
            results.append(doc)
        
        return results
    
    def run_rag_query_with_llm(self, query: str, limit: int = 3) -> str:
        """
        Ejecutar una consulta RAG completa con respuesta del LLM:
        1. Recuperar documentos relevantes
        2. Enviar documentos + consulta al LLM
        3. Obtener respuesta generada
        """
        # Recuperar documentos relevantes
        relevant_docs = self.rag_query(query, limit=limit)
        
        if not relevant_docs:
            return "No se encontraron documentos relevantes para tu consulta."
        
        # Preparar contexto para el LLM
        context = "Información relevante:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Documento {i}:\n"
            context += f"Título: {doc['title']}\n"
            
            if doc['summary']:
                context += f"Resumen: {doc['summary']}\n"
            
            # Incluir solo los primeros 1000 caracteres del cuerpo para no sobrecargar el contexto
            if doc['body']:
                truncated_body = doc['body'][:1000] + "..." if len(doc['body']) > 1000 else doc['body']
                context += f"Contenido: {truncated_body}\n"
            
            context += "\n---\n\n"
        
        # Construir prompt para OpenAI
        prompt = f"""Basándote en la siguiente información, responde a esta consulta de forma clara y concisa:

Consulta: {query}

{context}

Respuesta:"""
        
        # Llamar a OpenAI para generar respuesta
        try:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,  # Usando el modelo configurado en .env
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado que responde preguntas basándose únicamente en la información proporcionada."},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error al generar respuesta con OpenAI: {e}")
            return f"Error al procesar la consulta: {str(e)}"


# Función de ejemplo para procesar un archivo local (como el ejemplo proporcionado)
def process_local_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Procesar un archivo JSON local como ejemplo."""
    with open(file_path, 'r') as file:
        # Para el ejemplo proporcionado, parece ser un único documento en JSON
        publication = json.load(file)
        return [publication]

# Función principal
def main():
    # Inicializar pipeline
    pipeline = RAGPipeline()
    
    # Modo de ejemplo con archivo local
    example_file = 'paste.txt'
    if os.path.exists(example_file):
        print(f"Procesando archivo de ejemplo: {example_file}")
        publications = process_local_json_file(example_file)
    else:
        # En producción, usar la API
        print("Descargando publicaciones desde API...")
        publications = pipeline.fetch_publications(limit=50)
    
    # Extraer contenido relevante
    processed_documents = []
    for pub in publications:
        doc = pipeline.extract_relevant_content(pub)
        processed_documents.append(doc)
    
    # Indexar documentos
    pipeline.index_documents(processed_documents)
    
    # Ejemplo de consulta RAG
    query = "¿Qué es el Circularity Gap Report?"
    print(f"\nConsulta: {query}")
    
    # Obtener respuesta
    response = pipeline.run_rag_query_with_llm(query)
    print("\nRespuesta:")
    print(response)


if __name__ == "__main__":
    main()