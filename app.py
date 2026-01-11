import streamlit as st
from rag import retrieve_hybrid, is_low_confidence
from summarize import build_final_answer

st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")

st.title("NS Knowledge Assistant (RAG) — Portfolio MVP")
st.caption("Docs: docs_public/ (demo) • docs_private/ (local, not tracked)")

query = st.text_input("Ask a question", placeholder="e.g., What are the steps to borrow equipment?")
top_k = st.slider("Top K chunks", 3, 10, 5)

alpha = st.slider("Hybrid weight (vector vs keyword)", 0.0, 1.0, 0.65, 0.05)
st.caption("Higher = more semantic (vector). Lower = more exact keyword matching.")

if st.button("Search") and query.strip():
    with st.spinner("Retrieving..."):
        results = retrieve_hybrid(query.strip(), k=top_k, alpha=alpha)

    low_conf = is_low_confidence(query.strip(), results, threshold=0.25, min_coverage=0.5)

    st.subheader("Final Answer (Extracted from Evidence)")
    if low_conf:
        st.warning(
            "Low confidence: the current document set may not contain a clear answer. "
            "Showing the closest matches anyway."
        )

    borrow_steps, return_steps = build_final_answer(results)

    shown_any = False
    if borrow_steps:
        shown_any = True
        st.markdown("**Borrow**")
        for s in borrow_steps:
            st.write(f"- {s}")

    if return_steps:
        shown_any = True
        st.markdown("**Return**")
        for s in return_steps:
            st.write(f"- {s}")

    if not shown_any:
        st.write("No answer found from the current document set.")

    st.subheader("Cited Evidence")
    for i, r in enumerate(results, start=1):
        src = r["meta"]["source"]
        cid = r["meta"]["chunk_id"]
        with st.expander(
            f"[{i}] {src} (chunk {cid}) — hybrid score {r['score']:.3f} "
            f"(vec {r['vector_part']:.2f}, kw {r['keyword_part']:.2f})",
            expanded=(i == 1),
        ):
            st.write(r["text"])
