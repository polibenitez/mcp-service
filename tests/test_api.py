import os
import json
import unittest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Agregar el directorio raíz del proyecto al path de Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ahora podemos importar desde src
from src.api import app
class TestRagAPI(unittest.TestCase):
    """Test para la API REST del servicio MCP-RAG"""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas de API"""
        cls.client = TestClient(app)
        
        # Preparar archivo de prueba si no existe
        if not os.path.exists('test_data.json'):
            test_data = {
                "nid": [{"value": 999}],
                "uuid": [{"value": "test-api-uuid-999"}],
                "title": [{"value": "Documento de prueba API"}],
                "body": [
                    {
                        "value": "<p>Contenido de prueba para API</p>",
                        "format": "full_html",
                        "processed": "<p>Contenido de prueba para API</p>",
                        "summary": "Resumen del documento de prueba API"
                    }
                ],
                "created": [{"value": "2025-03-11T12:00:00+00:00"}]
            }
            
            with open('test_data.json', 'w') as f:
                json.dump(test_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Limpieza después de las pruebas de API"""
        # Limpiar archivos de prueba
        if os.path.exists('test_data.json'):
            os.unlink('test_data.json')
    
    def test_health_endpoint(self):
        """Probar el endpoint de salud de la API"""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
    
    def test_index_file_endpoint(self):
        """Probar el endpoint para indexar un archivo"""
        response = self.client.post(
            "/index/file",
            json={"file_path": "test_data.json"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
    
    def test_query_endpoint(self):
        """Probar el endpoint de consulta"""
        # Primero indexamos un archivo para tener datos
        self.client.post("/index/file", json={"file_path": "test_data.json"})
        
        # Esperamos un momento para asegurar que la indexación en segundo plano termine
        import time
        time.sleep(2)
        
        # Ahora hacemos la consulta
        response = self.client.post(
            "/query",
            json={"query": "documento de prueba", "limit": 3}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verificar estructura de respuesta
        self.assertIn("query", data)
        self.assertIn("answer", data)
        self.assertIn("documents", data)
    
    def test_query_without_documents(self):
        """Probar la consulta sin incluir documentos en la respuesta"""
        # Aseguramos que hay datos indexados
        self.client.post("/index/file", json={"file_path": "test_data.json"})
        
        # Esperamos un momento
        import time
        time.sleep(2)
        
        # Consulta sin incluir documentos
        response = self.client.post(
            "/query",
            json={"query": "documento de prueba", "include_documents": False}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verificar que no hay documentos en la respuesta
        self.assertEqual(data["documents"], [])
    
    def test_reset_endpoint(self):
        """Probar el endpoint para resetear la base de datos"""
        response = self.client.post("/reset")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")


if __name__ == "__main__":
    unittest.main()