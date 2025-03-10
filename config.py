import os
from dotenv import load_dotenv

def load_config():
    """
    Carga y valida la configuración desde el archivo .env
    Retorna un diccionario con la configuración
    """
    # Asegurar que se cargan las variables de entorno
    os.environ.clear()
    load_dotenv()
    print(os.getenv("EMBEDDING_MODEL"))
    # Configuración de OpenAI/LLM
    config = {
        # OpenAI o API compatible
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "llama-3.3-70b-instruct"),
        "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATUR", "0.3")),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-en"),
        
        # API de publicaciones
        "API_ENDPOINT": os.getenv("API_ENDPOINT"),
        
        # Qdrant
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "publications"),
        
        # Vectores
        "VECTOR_SIZE": 1536,  # Tamaño para el modelo text-embedding-ada-002

        # proxy
        "USER_PROXY": os.getenv("USER_PROXY", "benmanu"),
        "PASS_PROXY": os.getenv("PASS_PROXY", ""),
    }
    
    # Validar configuración crítica
    _validate_config(config)
    
    return config

def _validate_config(config):
    """Valida que la configuración mínima necesaria esté presente"""
    critical_keys = ["OPENAI_API_KEY"]
    
    missing_keys = [key for key in critical_keys if not config.get(key)]
    
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(f"Faltan variables de entorno críticas: {missing_keys_str}. "
                         f"Por favor, configura estas variables en el archivo .env")

# Uso del módulo
if __name__ == "__main__":
    try:
        config = load_config()
        print("Configuración cargada correctamente:")
        for key, value in config.items():
            # Ocultar la API key por seguridad
            if key == "OPENAI_API_KEY" and value:
                value = value[:4] + "..." + value[-4:]
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")