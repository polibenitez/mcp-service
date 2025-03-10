# Pipeline RAG con Qdrant y OpenAI

Este proyecto implementa un sistema de Retrieval-Augmented Generation (RAG) para procesar y consultar publicaciones web, vectorizando su contenido y almacenándolo en una base de datos Qdrant para su posterior recuperación.

## Descripción General

El sistema funciona como una tubería (pipeline) con las siguientes etapas:

1. **Extracción de datos**: Recupera publicaciones desde una API o archivo local
2. **Procesamiento**: Extrae el contenido relevante de cada publicación
3. **Vectorización**: Convierte el texto en embeddings usando OpenAI
4. **Almacenamiento**: Guarda los vectores y metadatos en Qdrant
5. **Consulta**: Permite realizar búsquedas semánticas y generar respuestas usando un LLM

## Analogía para entender el proceso

Para entender mejor cómo funciona este sistema RAG, piensa en él como una biblioteca inteligente:

- Las **publicaciones** son como libros con información valiosa
- El **procesamiento de texto** es como un bibliotecario que extrae las partes más importantes de cada libro (título, resumen, contenido)
- La **vectorización** es como crear una "huella digital" única para cada libro, que captura su significado
- **Qdrant** es como un sistema de estanterías inteligentes que organiza los libros según su similitud semántica
- La **consulta RAG** es como pedirle a un bibliotecario experto que:
  1. Busque los libros más relevantes para tu pregunta
  2. Lea rápidamente esos libros
  3. Te proporcione una respuesta basada exclusivamente en esa información

## Requisitos

- Python 3.8+
- Una API key de OpenAI
- Qdrant instalado localmente (o accesible en localhost:6333)

## Configuración

1. Clona este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Crea un archivo `.env` con las siguientes variables (usa el archivo `.env.example` como referencia):

```
# OpenAI o servicio compatible con la API de OpenAI
OPENAI_API_KEY=tu_clave_api_de_openai
OPENAI_API_BASE=https://api.openai.com/v1  # URL base para OpenAI u otro servicio compatible
LLM_MODEL=gpt-4-turbo  # Modelo para respuestas
LLM_TEMPERATURE=0.3  # Temperatura para controlar creatividad
EMBEDDING_MODEL=text-embedding-ada-002  # Modelo para embeddings

# API de publicaciones
API_ENDPOINT=url_de_tu_api_de_publicaciones

# Configuración de Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=publications
```

## Estructura del Código

El código se organiza en una clase principal `RAGPipeline` con los siguientes métodos:

- `fetch_publications`: Descarga publicaciones desde la API
- `extract_relevant_content`: Extrae información relevante de cada publicación
- `create_embedding`: Genera vectores a partir del texto usando OpenAI
- `index_documents`: Almacena documentos y sus vectores en Qdrant
- `rag_query`: Realiza consultas semánticas contra la base de datos vectorial
- `run_rag_query_with_llm`: Ejecuta una consulta completa RAG con respuesta del LLM

## Uso

```python
# Inicializar el pipeline
pipeline = RAGPipeline()

# Procesar documentos (desde API o archivo local)
publications = pipeline.fetch_publications(limit=50)  # o process_local_json_file()
processed_documents = [pipeline.extract_relevant_content(pub) for pub in publications]
pipeline.index_documents(processed_documents)

# Realizar una consulta RAG
respuesta = pipeline.run_rag_query_with_llm("¿Qué es el Circularity Gap Report?")
print(respuesta)
```

## Personalización

El código está diseñado para ser adaptable a diferentes formatos de datos. La función `extract_relevant_content` debe modificarse si el formato de tus publicaciones difiere del ejemplo proporcionado.

## Aspectos Técnicos Importantes

1. **Vectorización Eficiente**: Concatenamos título, resumen y cuerpo para crear un embedding que representa todo el documento.

2. **Manejo de Errores**: Se implementan medidas para gestionar textos vacíos o errores en la generación de embeddings.

3. **Metadatos Estructurados**: Además del contenido, almacenamos metadatos útiles como fechas, ubicaciones y organizaciones relacionadas.

4. **Consulta Contextualizada**: Al generar respuestas, proporcionamos al LLM contexto suficiente pero no excesivo (limitando el tamaño del texto).

## Optimizaciones Futuras

- Implementar chunking (división) de documentos largos para mejor recuperación
- Añadir filtrado por metadatos en las consultas
- Implementar reranking para mejorar la precisión
- Añadir caché para consultas frecuentes