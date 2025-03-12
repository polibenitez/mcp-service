from sentence_transformers import SentenceTransformer
# Load model directly
from transformers import AutoTokenizer, AutoModel
print("Transformer++++++++++++++++++++++++++++++++++++++")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# O guardar en una ruta específica
model_path = "./modelos/MiniLM-L6-v2"
model.save(model_path)

print(f"Modelo guardado en: {model_path}")