import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    import faiss
    faiss_index_path = os.path.join(INDEX_DIR, "docs.faiss")
    store_path = os.path.join(INDEX_DIR, "store.pkl")

    if not (os.path.exists(faiss_index_path) and os.path.exists(store_path)):
        raise RuntimeError("Index not found. Run python ingest.py first.")

    index = faiss.read_index(faiss_index_path)
    with open(store_path, "rb") as f:
        store = pickle.load(f)

    return index, store["chunks"], store["metas"]


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    index, chunks, metas = load_index()
    model = SentenceTransformer(EMB_MODEL_NAME)

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": chunks[idx],
            "meta": metas[idx],
        })
    return results
