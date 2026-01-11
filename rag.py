import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# simple module-level cache (so we don't reload every query)
_MODEL = None
_INDEX = None
_CHUNKS = None
_METAS = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMB_MODEL_NAME)
    return _MODEL


def _load_index_once():
    global _INDEX, _CHUNKS, _METAS

    if _INDEX is not None:
        return _INDEX, _CHUNKS, _METAS

    import faiss
    faiss_index_path = os.path.join(INDEX_DIR, "docs.faiss")
    store_path = os.path.join(INDEX_DIR, "store.pkl")

    if not (os.path.exists(faiss_index_path) and os.path.exists(store_path)):
        raise RuntimeError("Index not found. Run python ingest.py first.")

    _INDEX = faiss.read_index(faiss_index_path)

    with open(store_path, "rb") as f:
        store = pickle.load(f)

    _CHUNKS = store["chunks"]
    _METAS = store["metas"]
    return _INDEX, _CHUNKS, _METAS


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    index, chunks, metas = _load_index_once()
    model = _get_model()

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append(
            {
                "score": float(score),
                "text": chunks[idx],
                "meta": metas[idx],
            }
        )
    return results

def is_low_confidence(results: List[Dict[str, Any]], threshold: float = 0.25) -> bool:
    if not results:
        return True
    return results[0]["score"] < threshold
