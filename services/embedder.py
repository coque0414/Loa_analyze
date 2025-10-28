# services/embedder.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

_MODEL_NAME = "BM-K/KoSimCSE-roberta-multitask"

# 전역 캐시 (lazy load)
_tokenizer = None
_model = None
_device = None
_instance = None  # KoSimCSEEmbedder 단일톤

def _load_model():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModel.from_pretrained(_MODEL_NAME).to(_device).eval()

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # attention 가중 평균
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class KoSimCSEEmbedder:
    """
    SentenceTransformer 호환 encode(...) 메서드를 제공하는 래퍼.
    - inputs: str 또는 List[str]
    - returns: numpy.ndarray (convert_to_numpy=True) 또는 torch.Tensor
    - 옵션: normalize_embeddings=True/False
    """
    def __init__(self):
        _load_model()

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        batch_size: int = 32,
        dtype: np.dtype = np.float32,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        embs: List[np.ndarray] = []
        # 배치 인코딩
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = _tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True
            )
            inputs = {k: v.to(_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model(**inputs)
                sent_emb = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                if normalize_embeddings:
                    sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

            if convert_to_numpy:
                embs.append(sent_emb.detach().cpu().numpy().astype(dtype))
            else:
                embs.append(sent_emb)

        if convert_to_numpy:
            arr = np.vstack(embs)
            return arr if len(arr) > 1 else arr[0]
        else:
            tensor = torch.cat(embs, dim=0)
            return tensor if tensor.size(0) > 1 else tensor[0]

def get_embedder() -> KoSimCSEEmbedder:
    """기존 코드와 호환되는 객체(encode 메서드 제공)를 반환"""
    global _instance
    if _instance is None:
        _instance = KoSimCSEEmbedder()
    return _instance


# from sentence_transformers import SentenceTransformer

# _embedder = None

# def get_embedder():
#     global _embedder
#     if _embedder is None:
#         _embedder = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")
#     return _embedder
