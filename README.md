# RAG Knowledge Assistant (Streamlit + FAISS)

I built this as a simple RAG-style “knowledge search” tool: you put documents into a folder, ask questions, and it returns the most relevant excerpts with citations (source + chunk id).  
Goal is to keep it lightweight first, then improve it later.

## What it can do
- Read Markdown / TXT / PDF documents
- Split documents into overlapping chunks (so context won’t break)
- Convert chunks into embeddings (SentenceTransformers)
- Store embeddings in a FAISS index for fast semantic search
- Streamlit UI to query and show top-k cited evidence
- Shows a Final Answer extracted from top evidence
- Warns on low-confidence retrieval

## Tech used
- Python
- Streamlit
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS (vector search)
- PyPDF (PDF reader)

## Folder structure
rag-knowledge-assistant/
─ app.py # Streamlit UI
─ ingest.py # Build FAISS index from docs_public/
─ rag.py # Retrieval logic
─ requirements.txt
─ eval_public.md # Simple evaluation checklist
─ docs_public/ # Example docs for demo
─ docs_private/ # Local docs (ignored by git)


## How it works (high level)
1. Load docs from `docs_public/`
2. Chunk text with overlap
3. Embed each chunk into vectors
4. Save vectors into FAISS
5. For a query: embed query → retrieve nearest chunks → display results with citations

## Run locally (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python ingest.py
streamlit run app.py
