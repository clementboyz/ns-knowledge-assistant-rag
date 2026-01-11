import streamlit as st
from rag import retrieve

st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

st.title("NS Knowledge Assistant (RAG) — Portfolio MVP")
st.caption("Public docs in docs_public/. Restricted unit docs go in docs_private/ (gitignored).")

query = st.text_input("Ask a question", placeholder="e.g., What are the steps to borrow equipment?")
top_k = st.slider("Top K chunks", 3, 10, 5)

if st.button("Search") and query.strip():
    with st.spinner("Retrieving..."):
        results = retrieve(query.strip(), k=top_k)

    st.subheader("Answer (Retrieval-Only MVP)")
    st.write("Most relevant excerpts are shown below with citations (source + chunk id).")

    st.subheader("Cited Evidence")
    for i, r in enumerate(results, start=1):
        src = r["meta"]["source"]
        cid = r["meta"]["chunk_id"]
        with st.expander(f"[{i}] {src} (chunk {cid}) — score {r['score']:.3f}", expanded=(i == 1)):
            st.write(r["text"])
