import os
import pickle
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def retrieve_vector(query: str, k: int = 5) -> List[Dict[str, Any]]:
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
                "method": "vector",
            }
        )
    return results


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def retrieve_keyword(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Simple keyword scoring:
    score = (# of query tokens appearing in chunk) / (# of query tokens)
    """
    _, chunks, metas = _load_index_once()

    q_tokens = list(dict.fromkeys(_tokenize(query)))
    if not q_tokens:
        return []

    scored = []
    for i, chunk in enumerate(chunks):
        c_low = chunk.lower()
        hit = sum(1 for t in q_tokens if t in c_low)
        score = hit / max(len(q_tokens), 1)
        if score > 0:
            scored.append((score, i))

    scored.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, idx in scored[:k]:
        results.append(
            {
                "score": float(score),
                "text": chunks[idx],
                "meta": metas[idx],
                "method": "keyword",
            }
        )
    return results


def retrieve_hybrid(query: str, k: int = 5, alpha: float = 0.65) -> List[Dict[str, Any]]:
    """
    Combine vector + keyword results.
    alpha controls weight of vector score (0..1).
    """
    vec = retrieve_vector(query, k=max(k, 10))
    kw = retrieve_keyword(query, k=max(k, 10))

    # normalize scores within each method to 0..1
    def norm(res: List[Dict[str, Any]]) -> Dict[Tuple[str, int], float]:
        if not res:
            return {}
        s = np.array([r["score"] for r in res], dtype=float)
        smin, smax = float(s.min()), float(s.max())
        out = {}
        for r in res:
            key = (r["meta"]["source"], r["meta"]["chunk_id"])
            if smax - smin < 1e-9:
                out[key] = 1.0
            else:
                out[key] = (r["score"] - smin) / (smax - smin)
        return out

    vec_n = norm(vec)
    kw_n = norm(kw)

    # merge keys
    all_keys = set(vec_n.keys()) | set(kw_n.keys())

    # build a lookup for text/meta
    lookup = {}
    for r in vec + kw:
        key = (r["meta"]["source"], r["meta"]["chunk_id"])
        lookup[key] = r

    merged = []
    for key in all_keys:
        v = vec_n.get(key, 0.0)
        w = kw_n.get(key, 0.0)
        score = alpha * v + (1 - alpha) * w

        r = lookup[key]
        merged.append(
            {
                "score": float(score),
                "text": r["text"],
                "meta": r["meta"],
                "method": "hybrid",
                "vector_part": float(v),
                "keyword_part": float(w),
            }
        )

    merged.sort(reverse=True, key=lambda x: x["score"])
    return merged[:k]


def is_low_confidence(results: List[Dict[str, Any]], threshold: float = 0.25) -> bool:
    if not results:
        return True
    return results[0]["score"] < threshold
