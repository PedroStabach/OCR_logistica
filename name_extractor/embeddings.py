import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

def load_or_build_embeddings(motoristas):
    if not _HAS_ST:
        return None, None

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(motoristas, convert_to_numpy=True, normalize_embeddings=True)
    return model, emb
