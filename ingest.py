import os
import glob
import pickle
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

INDEX_DIR = "index"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    # keep newlines reasonably, but remove excessive whitespace
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks


def load_docs(doc_dir: str) -> List[Tuple[str, str]]:
    files = []
    files += glob.glob(os.path.join(doc_dir, "**/*.md"), recursive=True)
    files += glob.glob(os.path.join(doc_dir, "**/*.txt"), recursive=True)
    files += glob.glob(os.path.join(doc_dir, "**/*.pdf"), recursive=True)

    docs = []
    for fp in files:
        if fp.lower().endswith(".pdf"):
            content = read_pdf(fp)
        else:
            content = read_text_file(fp)
        if content.strip():
            docs.append((fp.replace("\\", "/"), content))
    return docs


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_docs("docs_public")
    if not docs:
        raise RuntimeError("No documents found in docs_public/")

    model = SentenceTransformer(EMB_MODEL_NAME)

    chunks = []
    metadatas = []
    for path, text in docs:
        for i, c in enumerate(chunk_text(text)):
            chunks.append(c)
            metadatas.append({"source": path, "chunk_id": i})

    emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.faiss"))
    with open(os.path.join(INDEX_DIR, "store.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks, "metas": metadatas}, f)

    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    print("Done. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
