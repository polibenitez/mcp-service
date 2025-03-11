from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional
import json
import os

from mcp_architecture import MCPRagService

# Inicialización del servicio
rag_service = MCPRagService()

# Modelos Pydantic para validación
class QueryRequest(BaseModel):
    query: str
    limit: int = 3
    include_documents: bool = True

class IndexFileRequest(BaseModel):
    file_path: str

class IndexApiRequest(BaseModel):
    limit: int = 100

class Document(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: Optional[List[Document]] = None

class StatusResponse(BaseModel):
    status: str
    details: Optional[str] = None

# Aplicación FastAPI
app = FastAPI(
    title="MCP-RAG API Service",
    description="API para el servicio RAG basado en Model-Context-Protocol",
    version="1.0.0"
)

# Configurar CORS para permitir acceso desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Realiza una consulta RAG y devuelve la respuesta generada.
    
    - **query**: El texto de la consulta
    - **limit**: Número máximo de documentos a recuperar (default: 3)
    - **include_documents**: Si se incluyen los documentos en la respuesta (default: true)
    """
    try:
        result = rag_service.query(request.query, request.limit)
        
        # Si no se solicitan documentos, no los incluimos en la respuesta
        if not request.include_documents:
            result["documents"] = []
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta: {str(e)}")

@app.post("/index/file", response_model=StatusResponse)
async def index_file(request: IndexFileRequest, background_tasks: BackgroundTasks):
    """
    Indexa publicaciones desde un archivo local.
    
    - **file_path**: Ruta al archivo JSON con los datos
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {request.file_path}")
    
    try:
        # Ejecutar la indexación en segundo plano para no bloquear la respuesta
        background_tasks.add_task(rag_service.index_publication_from_file, request.file_path)
        return {"status": "success", "details": f"Indexación de {request.file_path} iniciada en segundo plano"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar archivo: {str(e)}")

@app.post("/index/api", response_model=StatusResponse)
async def index_api(request: IndexApiRequest, background_tasks: BackgroundTasks):
    """
    Indexa publicaciones desde la API configurada.
    
    - **limit**: Número máximo de publicaciones a indexar (default: 100)
    """
    try:
        # Ejecutar la indexación en segundo plano
        background_tasks.add_task(rag_service.index_publications_from_api, request.limit)
        return {"status": "success", "details": f"Indexación de {request.limit} publicaciones iniciada en segundo plano"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar desde API: {str(e)}")

@app.post("/reset", response_model=StatusResponse)
async def reset_database():
    """
    Resetea la base de datos vectorial.
    """
    try:
        success = rag_service.reset_database()
        if not success:
            raise Exception("Error interno al resetear la base de datos")
        return {"status": "success", "details": "Base de datos reiniciada correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resetear la base de datos: {str(e)}")

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """
    Endpoint para verificar el estado del servicio.
    """
    return {"status": "ok"}

# Punto de entrada para ejecutar la aplicación
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)