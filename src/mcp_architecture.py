# mcp_architecture.py
"""
Implementación de Model-Context-Protocol (MCP) para el sistema RAG
 
El MCP separa las responsabilidades en tres componentes:
1. Model: Encapsula la lógica de procesamiento (LLM, embeddings)
2. Context: Gestiona los datos y el estado (vectores, documentos)
3. Protocol: Maneja la comunicación entre componentes y servicios externos
"""

import json
import requests
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import load_config

# Cargar configuración
config = load_config()

# ========================
# COMPONENT: MODEL
# ========================
class ModelComponent:
    """
    Componente Model: Responsable de la lógica de procesamiento 
    como generación de embeddings y respuestas del LLM
    """
    
    def __init__(self, openai_client, config):
        self.openai_client = openai_client
        self.config = config
    
    def create_embedding(self, text: str) -> List[float]:
        """Crear un embedding para el texto usando el modelo configurado"""
        if not text.strip():
            return np.zeros(self.config["VECTOR_SIZE"]).tolist()
            
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.config["EMBEDDING_MODEL"]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return np.zeros(self.config["VECTOR_SIZE"]).tolist()
    
    def create_document_embedding(self, document: Dict[str, Any]) -> List[float]:
        """Crear embedding para un documento completo"""
        combined_text = document["title"]
        
        if document.get("summary"):
            combined_text += " " + document["summary"]
        
        if document.get("body"):
            combined_text += " " + document["body"]
        
        return self.create_embedding(combined_text)
    
    def generate_response(self, 
                         query: str, 
                         context_docs: List[Dict[str, Any]]) -> str:
        """Generar respuesta del LLM basada en el contexto recuperado"""
        # Preparar contexto para el LLM
        context = "Información relevante:\n\n"
        
        for i, doc in enumerate(context_docs, 1):
            context += f"Documento {i}:\n"
            context += f"Título: {doc['title']}\n"
            
            if doc.get('summary'):
                context += f"Resumen: {doc['summary']}\n"
            
            if doc.get('body'):
                truncated_body = doc['body'][:1000] + "..." if len(doc['body']) > 1000 else doc['body']
                context += f"Contenido: {truncated_body}\n"
            
            context += "\n---\n\n"
        
        # Construir prompt para OpenAI
        prompt = f"""Basándote en la siguiente información, responde a esta consulta de forma clara y concisa:

Consulta: {query}

{context}

Respuesta:"""
        
        # Llamar al LLM para generar respuesta
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado que responde preguntas basándose únicamente en la información proporcionada."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config["LLM_TEMPERATURE"]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error al generar respuesta con LLM: {e}")
            return f"Error al procesar la consulta: {str(e)}"

# ========================
# COMPONENT: CONTEXT
# ========================
class ContextComponent:
    """
    Componente Context: Responsable de gestionar los datos y el estado,
    incluyendo el almacenamiento y recuperación de vectores y documentos
    """
    
    def __init__(self, qdrant_client, config):
        self.qdrant_client = qdrant_client
        self.config = config
        self._setup_collection()
    
    def _setup_collection(self):
        """Configurar la colección en Qdrant si no existe"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.config["COLLECTION_NAME"] not in collection_names:
            print(f"Creando colección {self.config['COLLECTION_NAME']}...")
            self.qdrant_client.create_collection(
                collection_name=self.config["COLLECTION_NAME"],
                vectors_config=models.VectorParams(
                    size=self.config["VECTOR_SIZE"], 
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            print(f"Colección {self.config['COLLECTION_NAME']} creada correctamente")
    
    def store_document(self, doc_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        """Almacenar un documento en la base de datos vectorial"""
        point = models.PointStruct(
            id=doc_id,
            vector=vector,
            payload=payload
        )
        
        self.qdrant_client.upsert(
            collection_name=self.config["COLLECTION_NAME"],
            points=[point]
        )
    
    def store_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]) -> None:
        """Almacenar múltiples documentos en lote"""
        if not documents or not vectors or len(documents) != len(vectors):
            raise ValueError("La lista de documentos y vectores debe tener la misma longitud")
        
        points = []
        for i, doc in enumerate(documents):
            doc_id = doc["id"] if isinstance(doc["id"], str) else str(doc["id"])
            
            # Preparar payload para almacenamiento
            payload = {
                "title": doc["title"],
                "summary": doc.get("summary", ""),
                "body": doc.get("body", ""),
                "metadata": doc.get("metadata", {})
            }
            
            point = models.PointStruct(
                id=doc_id,
                vector=vectors[i],
                payload=payload
            )
            
            points.append(point)
        
        # Realizar la inserción por lotes
        if points:
            self.qdrant_client.upsert(
                collection_name=self.config["COLLECTION_NAME"],
                points=points
            )
    
    def retrieve_documents(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Recuperar documentos basados en similitud vectorial"""
        search_result = self.qdrant_client.search(
            collection_name=self.config["COLLECTION_NAME"],
            query_vector=query_vector,
            limit=limit
        )
        
        # Preparar resultado
        results = []
        for scored_point in search_result:
            doc = {
                "id": scored_point.id,
                "score": scored_point.score,
                "title": scored_point.payload.get("title", ""),
                "summary": scored_point.payload.get("summary", ""),
                "body": scored_point.payload.get("body", ""),
                "metadata": scored_point.payload.get("metadata", {})
            }
            results.append(doc)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Recuperar un documento específico por su ID"""
        try:
            result = self.qdrant_client.retrieve(
                collection_name=self.config["COLLECTION_NAME"],
                ids=[doc_id]
            )
            
            if result and len(result) > 0:
                point = result[0]
                return {
                    "id": point.id,
                    "title": point.payload.get("title", ""),
                    "summary": point.payload.get("summary", ""),
                    "body": point.payload.get("body", ""),
                    "metadata": point.payload.get("metadata", {})
                }
            return None
        except Exception as e:
            print(f"Error recuperando documento {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Eliminar un documento de la base de datos"""
        try:
            self.qdrant_client.delete(
                collection_name=self.config["COLLECTION_NAME"],
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error eliminando documento {doc_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Limpiar toda la colección (útil para pruebas)"""
        try:
            self.qdrant_client.delete_collection(
                collection_name=self.config["COLLECTION_NAME"]
            )
            self._setup_collection()
            return True
        except Exception as e:
            print(f"Error limpiando colección: {e}")
            return False

# ========================
# COMPONENT: PROTOCOL
# ========================
class ProtocolComponent:
    """
    Componente Protocol: Responsable de la comunicación entre componentes
    y servicios externos, como APIs de publicaciones
    """
    
    def __init__(self, config):
        self.config = config
        
    def fetch_publications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener publicaciones desde la API externa"""
        print("Descargando publicaciones desde la API...")
        
        try:
            response = requests.get(f"{self.config['API_ENDPOINT']}?limit={limit}")
            
            if response.status_code == 200:
                publications = response.json()
                print(f"Se descargaron {len(publications)} publicaciones")
                return publications
            else:
                raise Exception(f"Error al obtener publicaciones: {response.status_code}")
        except Exception as e:
            print(f"Error en la comunicación con la API: {e}")
            return []
    
    def process_local_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Procesar un archivo JSON local como fuente de datos"""
        try:
            with open(file_path, 'r') as file:
                publication = json.load(file)
                return [publication]
        except Exception as e:
            print(f"Error procesando archivo local: {e}")
            return []
    
    def extract_relevant_content(self, publication: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer contenido relevante de una publicación"""
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

# ========================
# MCP FACADE
# ========================
class MCPRagService:
    """
    Fachada principal para el sistema RAG usando arquitectura MCP.
    Orquesta los componentes Model, Context y Protocol.
    """
    
    def __init__(self):
        # Cargar configuración
        self.config = load_config()
        
        # Configurar cliente OpenAI con base URL opcional
        openai_client_kwargs = {"api_key": self.config["OPENAI_API_KEY"]}
        if self.config["OPENAI_API_BASE"]:
            openai_client_kwargs["base_url"] = self.config["OPENAI_API_BASE"]
        
        # Inicializar clientes
        self.openai_client = OpenAI(**openai_client_kwargs)
        self.qdrant_client = QdrantClient(
            host=self.config["QDRANT_HOST"], 
            port=self.config["QDRANT_PORT"]
        )
        
        # Inicializar componentes MCP
        self.model = ModelComponent(self.openai_client, self.config)
        self.context = ContextComponent(self.qdrant_client, self.config)
        self.protocol = ProtocolComponent(self.config)
    
    def index_publications_from_api(self, limit: int = 100) -> int:
        """Indexar publicaciones desde la API"""
        # Protocolo: Obtener publicaciones
        publications = self.protocol.fetch_publications(limit)
        
        # Protocolo: Extraer contenido relevante
        processed_documents = []
        vectors = []
        
        for pub in publications:
            doc = self.protocol.extract_relevant_content(pub)
            processed_documents.append(doc)
            
            # Modelo: Generar embeddings
            vector = self.model.create_document_embedding(doc)
            vectors.append(vector)
        
        # Contexto: Almacenar documentos
        self.context.store_documents(processed_documents, vectors)
        
        return len(processed_documents)
    
    def index_publication_from_file(self, file_path: str) -> int:
        """Indexar publicaciones desde un archivo local"""
        # Protocolo: Obtener publicaciones
        publications = self.protocol.process_local_json_file(file_path)
        
        # Protocolo: Extraer contenido relevante
        processed_documents = []
        vectors = []
        
        for pub in publications:
            doc = self.protocol.extract_relevant_content(pub)
            processed_documents.append(doc)
            
            # Modelo: Generar embeddings
            vector = self.model.create_document_embedding(doc)
            vectors.append(vector)
        
        # Contexto: Almacenar documentos
        self.context.store_documents(processed_documents, vectors)
        
        return len(processed_documents)
    
    def query(self, query_text: str, limit: int = 5) -> Dict[str, Any]:
        """
        Realizar una consulta completa:
        1. Modelo: Vectorizar consulta
        2. Contexto: Recuperar documentos relevantes
        3. Modelo: Generar respuesta con LLM
        """
        # Modelo: Vectorizar consulta
        query_vector = self.model.create_embedding(query_text)
        
        # Contexto: Recuperar documentos relevantes
        relevant_docs = self.context.retrieve_documents(query_vector, limit)
        
        if not relevant_docs:
            return {
                "query": query_text,
                "documents": [],
                "answer": "No se encontraron documentos relevantes para tu consulta."
            }
        
        # Modelo: Generar respuesta
        answer = self.model.generate_response(query_text, relevant_docs)
        
        # Preparar resultado
        return {
            "query": query_text,
            "documents": relevant_docs,
            "answer": answer
        }
    
    def reset_database(self) -> bool:
        """Limpiar la base de datos vectorial"""
        return self.context.clear_collection()


# ============================
# API SERVICE (opcional para exponer como servicio REST)
# ============================
"""
Para exponer el servicio MCP-RAG como una API REST, puedes utilizar FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="MCP-RAG API")
service = MCPRagService()

class QueryRequest(BaseModel):
    query: str
    limit: int = 3

@app.post("/query")
async def query(request: QueryRequest):
    result = service.query(request.query, request.limit)
    return result

@app.post("/index/file")
async def index_file(file_path: str):
    count = service.index_publication_from_file(file_path)
    return {"indexed_count": count}

@app.post("/index/api")
async def index_api(limit: int = 100):
    count = service.index_publications_from_api(limit)
    return {"indexed_count": count}

@app.post("/reset")
async def reset_database():
    success = service.reset_database()
    if not success:
        raise HTTPException(500, "Error al resetear la base de datos")
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```
"""


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar el servicio MCP-RAG
    rag_service = MCPRagService()
    
    # Procesar archivo de ejemplo
    example_file = 'paste.txt'
    print(f"Indexando archivo: {example_file}")
    indexed_count = rag_service.index_publication_from_file(example_file)
    print(f"Se indexaron {indexed_count} documentos")
    
    # Realizar consulta
    query_text = "¿Qué es el Circularity Gap Report?"
    print(f"\nConsulta: {query_text}")
    
    result = rag_service.query(query_text)
    
    print("\nRespuesta:")
    print(result["answer"])
    
    print("\nDocumentos relevantes:")
    for i, doc in enumerate(result["documents"], 1):
        print(f"{i}. {doc['title']} (Similitud: {doc['score']:.2f})")