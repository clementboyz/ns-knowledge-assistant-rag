# RAG Knowledge Assistant (Streamlit + FAISS)

A lightweight Retrieval-Augmented pipeline that retrieves relevant excerpts from a document folder and displays results with citations (source + chunk id).

## Features
- Ingests Markdown/TXT/PDF documents
- Splits documents into overlapping chunks
- Embeds chunks with SentenceTransformers
- Stores embeddings in a FAISS index for fast semantic search
- Streamlit UI to query and view top-k cited evidence

## Tech Stack
- Python
- Streamlit
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS (vector search)
- PyPDF (PDF reading)

## Project Structure
rag-knowledge-assistant/
├─ app.py # Streamlit UI
├─ ingest.py # Build FAISS index from docs_public/
├─ rag.py # Retrieval logic
├─ requirements.txt
├─ eval_public.md # Simple evaluation checklist
├─ docs_public/ # Example documents used for demo
└─ docs_private/ # Local documents (ignored by git)


## How it works
1. Read documents from `docs_public/`
2. Chunk text with overlap to preserve context
3. Generate embeddings for each chunk
4. Save embeddings into a FAISS index
5. For a user query: embed query → retrieve nearest chunks → display cited excerpts

## Setup & Run
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python ingest.py
streamlit run app.py
