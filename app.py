import streamlit as st
from rag import retrieve, is_low_confidence
from summarize import build_final_answer


st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

st.title("NS Knowledge Assistant (RAG) — Portfolio MVP")
st.caption("Public docs in docs_public/. Restricted unit docs go in docs_private/ (gitignored).")

query = st.text_input("Ask a question", placeholder="e.g., What are the steps to borrow equipment?")
top_k = st.slider("Top K chunks", 3, 10, 5)

if st.button("Search") and query.strip():
    with st.spinner("Retrieving..."):
        results = retrieve(query.strip(), k=top_k)

    low_conf = is_low_confidence(results, threshold=0.25)

    st.subheader("Final Answer (Extracted from Evidence)")
    if low_conf:
        st.warning("Low confidence: the documents may not contain a clear answer. Showing closest matches anyway.")
    borrow_steps, return_steps = build_final_answer(results)
    if borrow_steps:
        st.markdown("**Borrow**")
        for s in borrow_steps:
            st.write(f"- {s}")
    if return_steps:
        st.markdown("**Return**")
        for s in return_steps:
            st.write(f"- {s}")
    if not borrow_steps and not return_steps:
        st.write("No answer found from the current document set.")

    st.subheader("Cited Evidence")
    for i, r in enumerate(results, start=1):
        src = r["meta"]["source"]
        cid = r["meta"]["chunk_id"]
        with st.expander(f"[{i}] {src} (chunk {cid}) — score {r['score']:.3f}", expanded=(i == 1)):
            st.write(r["text"])
