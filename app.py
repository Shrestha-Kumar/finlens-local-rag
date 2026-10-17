"""Minimal web UI:  streamlit run app.py"""
import streamlit as st

from rag.generator import format_source
from rag.index import VectorIndex
from rag.pipeline import answer_question

st.set_page_config(page_title="Local Financial RAG", layout="wide")
st.title("Local Financial Document Q&A")
st.caption("Everything runs on this machine: local embeddings, FAISS, and an Ollama LLM.")


@st.cache_resource
def get_index():
    return VectorIndex.load()


try:
    index = get_index()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

question = st.text_input("Ask a question about your documents")
col1, col2 = st.columns(2)
k = col1.slider("Chunks to retrieve", 1, 10, 5)
retrieve_only = col2.checkbox("Retrieve only (skip the LLM)")

if question:
    try:
        with st.spinner("Searching..."):
            result = answer_question(index, question, k, retrieve_only)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    if result["answer"]:
        st.subheader("Answer")
        st.write(result["answer"])
    st.subheader("Sources")
    for i, c in enumerate(result["sources"], start=1):
        with st.expander(f"[{i}] {format_source(c)}  (score {c['score']:.3f})"):
            st.write(c["text"])
