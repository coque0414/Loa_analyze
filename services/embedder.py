from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")
    return _embedder