import os
import unittest
import json
import tempfile
from pathlib import Path
import sys

# Agregar el directorio raíz del proyecto al path de Python
# Esto permite importar módulos desde src sin instalar el paquete
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ahora podemos importar desde src
from src.mcp_architecture import MCPRagService, ModelComponent, ContextComponent, ProtocolComponent
from config import load_config


class TestMCPRagComponents(unittest.TestCase):
    """Pruebas unitarias para los componentes individuales del MCP RAG"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas"""
        # Cargar configuración
        cls.config = load_config()
        
        # Inicializar el servicio completo para pruebas
        cls.rag_service = MCPRagService()
        
        # Crear un documento de prueba
        cls.test_doc = {
            "id": "test123",
            "title": "Documento de prueba para RAG",
            "summary": "Este es un resumen del documento de prueba para verificar el sistema RAG",
            "body": "Este es el cuerpo principal del documento de prueba. Contiene información sobre pruebas de sistemas RAG y arquitectura MCP.",
            "metadata": {
                "created_date": "2025-03-11T12:00:00"
            }
        }
        
        # Crear un archivo temporal para pruebas
        cls.temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        cls.temp_filename = cls.temp_file.name
        
        # Escribir datos de ejemplo en el archivo temporal
        example_data = {
            "nid": [{"value": 12345}],
            "uuid": [{"value": "test-uuid-12345"}],
            "title": [{"value": "Documento de prueba desde archivo"}],
            "body": [
                {
                    "value": "<p>Contenido de prueba del documento</p>",
                    "format": "full_html",
                    "processed": "<p>Contenido de prueba del documento</p>",
                    "summary": "Resumen del documento de prueba desde archivo"
                }
            ],
            "created": [{"value": "2025-03-11T12:00:00+00:00"}]
        }
        
        json.dump(example_data, cls.temp_file)
        cls.temp_file.close()
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todas las pruebas"""
        # Eliminar el archivo temporal
        if os.path.exists(cls.temp_filename):
            os.unlink(cls.temp_filename)
        
        # Limpiar la colección de prueba en Qdrant
        cls.rag_service.reset_database()
    
    def test_model_component_embedding(self):
        """Probar la generación de embeddings del componente Model"""
        # Acceder al componente Model
        model = self.rag_service.model
        
        # Probar la creación de embeddings
        text = "Este es un texto de prueba para embeddings"
        embedding = model.create_embedding(text)
        
        # Verificar que el embedding tenga la dimensión correcta
        self.assertEqual(len(embedding), self.config["VECTOR_SIZE"])
        
        # Verificar que el embedding no sea un vector de ceros
        self.assertFalse(all(e == 0 for e in embedding))
    
    def test_model_component_document_embedding(self):
        """Probar la generación de embeddings para documentos completos"""
        # Acceder al componente Model
        model = self.rag_service.model
        
        # Probar la creación de embeddings para un documento
        embedding = model.create_document_embedding(self.test_doc)
        
        # Verificar que el embedding tenga la dimensión correcta
        self.assertEqual(len(embedding), self.config["VECTOR_SIZE"])
    
    def test_context_component_store_retrieve(self):
        """Probar el almacenamiento y recuperación de documentos"""
        # Acceder al componente Context
        context = self.rag_service.context
        
        # Crear un embedding para el documento de prueba
        vector = self.rag_service.model.create_document_embedding(self.test_doc)
        
        # Almacenar el documento
        context.store_document(
            doc_id=self.test_doc["id"],
            vector=vector,
            payload={
                "title": self.test_doc["title"],
                "summary": self.test_doc["summary"],
                "body": self.test_doc["body"],
                "metadata": self.test_doc["metadata"]
            }
        )
        
        # Recuperar el documento por ID
        retrieved_doc = context.get_document_by_id(self.test_doc["id"])
        
        # Verificar que se recuperó correctamente
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc["id"], self.test_doc["id"])
        self.assertEqual(retrieved_doc["title"], self.test_doc["title"])
    
    def test_protocol_component_extract_content(self):
        """Probar la extracción de contenido relevante de publicaciones"""
        # Acceder al componente Protocol
        protocol = self.rag_service.protocol
        
        # Crear datos de prueba similares a los de la API
        test_publication = {
            "nid": [{"value": 54321}],
            "title": [{"value": "Título de prueba"}],
            "body": [
                {
                    "processed": "<p>Contenido procesado</p>",
                    "summary": "Resumen de prueba"
                }
            ]
        }
        
        # Extraer contenido relevante
        extracted = protocol.extract_relevant_content(test_publication)
        
        # Verificar extracción
        self.assertEqual(extracted["id"], "54321")
        self.assertEqual(extracted["title"], "Título de prueba")
        self.assertEqual(extracted["summary"], "Resumen de prueba")
        self.assertEqual(extracted["body"], "<p>Contenido procesado</p>")
    
    def test_protocol_component_process_file(self):
        """Probar el procesamiento de archivos locales"""
        # Acceder al componente Protocol
        protocol = self.rag_service.protocol
        
        # Procesar el archivo temporal
        publications = protocol.process_local_json_file(self.temp_filename)
        
        # Verificar que se procesó correctamente
        self.assertEqual(len(publications), 1)
        self.assertIn("title", publications[0])
    
    def test_integrated_index_query(self):
        """Prueba de integración: indexar y consultar"""
        # Resetear la base de datos para esta prueba
        self.rag_service.reset_database()
        
        # Indexar el archivo de prueba
        count = self.rag_service.index_publication_from_file(self.temp_filename)
        self.assertEqual(count, 1)
        
        # Realizar una consulta
        query = "documento de prueba"
        result = self.rag_service.query(query, limit=1)
        
        # Verificar el resultado
        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertIn("documents", result)
        self.assertEqual(result["query"], query)
        self.assertTrue(len(result["documents"]) > 0)


class TestMCPRagService(unittest.TestCase):
    """Pruebas de integración para el servicio completo"""
    
    def setUp(self):
        """Configuración para cada prueba"""
        self.rag_service = MCPRagService()
        # Limpiar la base de datos antes de cada prueba
        self.rag_service.reset_database()
    
    def test_index_and_query_workflow(self):
        """Prueba del flujo completo: indexar archivo y realizar consulta"""
        # Verificar si existe el archivo paste.txt
        if not os.path.exists('paste.txt'):
            self.skipTest("Archivo paste.txt no encontrado para la prueba")
        
        # Indexar archivo de ejemplo
        count = self.rag_service.index_publication_from_file('paste.txt')
        self.assertGreater(count, 0)
        
        # Realizar una consulta
        result = self.rag_service.query("¿Qué es el Circularity Gap Report?")
        
        # Verificar estructura de la respuesta
        self.assertIn("answer", result)
        self.assertIn("documents", result)
        self.assertGreater(len(result["documents"]), 0)
        
        # Verificar que la respuesta no está vacía
        self.assertNotEqual(result["answer"].strip(), "")


if __name__ == '__main__':
    unittest.main()