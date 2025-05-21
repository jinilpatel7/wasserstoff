import os
import sys
import streamlit as st
import pandas as pd
import torch  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from backend.app.modules.upload import save_uploaded_files
from langchain_core.documents import Document
from backend.app.modules.text_extractor import extract_all_text
from backend.app.modules.ocr_processor import process_images
from backend.app.modules.embedder import Embedder
from backend.app.modules.query_engine import QueryEngine

st.set_page_config(page_title="AI Document Research Bot", layout="wide")
st.title("📄 Document Research & Theme Identifier")

DOCS_FOLDER = "./backend/data/uploads"
embedder = Embedder()
query_engine = QueryEngine()

# Init session state
if "extracted_docs" not in st.session_state:
    st.session_state.extracted_docs = {}
if "document_names" not in st.session_state:
    st.session_state.document_names = set()

# --- Upload ---
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs and images",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Saving and processing documents..."):
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        saved_paths = save_uploaded_files(uploaded_files)

        pdf_files = [f for f in saved_paths if f.lower().endswith(".pdf")]
        image_files = [f for f in saved_paths if not f.lower().endswith(".pdf")]

        ocr_texts = process_images(image_files) if image_files else {}
        pdf_texts = extract_all_text(pdf_files) if pdf_files else {}

        combined = {**pdf_texts, **ocr_texts}
        st.session_state.extracted_docs.update(combined)
        st.session_state.document_names.update([os.path.basename(f) for f in combined.keys()])

        if combined:
            embedder.store_embeddings(combined)
            st.success(f"Successfully processed {len(combined)} documents!")

# --- Query ---
if st.session_state.extracted_docs:
    st.sidebar.header("2. Query Documents")
    query = st.sidebar.text_area("Ask a question", placeholder="e.g. What are the penalties described in the documents?")

    if st.sidebar.button("Submit Query") and query.strip():
        with st.spinner("Answering your query per document..."):
            answers = []
            for name, text in st.session_state.extracted_docs.items():
                doc = Document(page_content=text, metadata={"source": name})
                try:
                    answer = query_engine.answer_query_single_document(query, doc)
                    answers.append({"document": name, "answer": answer})
                except Exception as e:
                    answers.append({"document": name, "answer": f"Error processing this document: {e}"})

            st.markdown("### 📝 Answers per Document")
            for entry in answers:
                st.markdown(f"**Document: {entry['document']}**")
                st.markdown(entry["answer"], unsafe_allow_html=True)
                st.markdown("---")

        with st.spinner("Identifying cross-document themes..."):
            all_docs = [
                Document(page_content=text, metadata={"source": name})
                for name, text in st.session_state.extracted_docs.items()
            ]
            themes = query_engine.identify_themes(all_docs)

            st.markdown("### 🎯 Identified Themes")
            if themes:
                for theme in themes:
                    st.markdown(f"**{theme['theme']}**")
                    st.markdown(theme["summary"])
                    if theme.get("supporting_docs"):
                        unique_docs = sorted(set(theme["supporting_docs"]))
                        st.markdown("**📚 Documents involved:** " + ", ".join(unique_docs))
            else:
                st.info("No themes could be extracted.")

# --- Always Show Uploaded Docs ---
if st.session_state.document_names:
    st.markdown("---")
    st.markdown("### 📂 Uploaded Documents")
    df_docs = pd.DataFrame({"Document Name": sorted(st.session_state.document_names)})
    st.dataframe(df_docs, use_container_width=True)
