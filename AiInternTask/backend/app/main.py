import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from modules.upload import save_uploaded_files
from langchain_core.documents import Document

from modules.text_extractor import extract_all_text
from modules.ocr_processor import process_images
from modules.embedder import Embedder
from modules.query_engine import QueryEngine

st.set_page_config(page_title="AI Document Research Bot", layout="wide")
st.title("📄 Document Research & Theme Identifier")

DOCS_FOLDER = "AiInternTask/backend/data/uploads"
embedder = Embedder()
query_engine = QueryEngine()

# Init session state
if "extracted_docs" not in st.session_state:
    st.session_state.extracted_docs = {}
if "document_names" not in st.session_state:
    st.session_state.document_names = []

# --- Upload ---
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs and image scans (75+ supported)",
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
        st.session_state.document_names.extend([os.path.basename(f) for f in combined.keys()])

        if combined:
            embedder.store_embeddings(combined)
            st.success(f"Successfully processed {len(combined)} documents!")

# --- Query ---
if st.session_state.extracted_docs:
    st.sidebar.header("2. Query Documents")
    query = st.sidebar.text_area("Ask a question", placeholder="e.g. What are the penalties described in the documents?")

    if st.sidebar.button("Submit Query") and query.strip():
        with st.spinner("Answering your query..."):
            result = query_engine.answer_query(query)

            st.markdown("### 📝 Answer")
            st.markdown(result["answer"], unsafe_allow_html=True)

            st.markdown("### 📑 Citations")
            df = pd.DataFrame(result["citations"])
            st.dataframe(df, use_container_width=True)

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
                        st.markdown("**📚 Documents involved:** " + ", ".join(theme["supporting_docs"]))
            else:
                st.info("No themes could be extracted.")

# --- Always Show Uploaded Docs ---
if st.session_state.document_names:
    st.markdown("---")
    st.markdown("### 📂 Uploaded Documents")
    df_docs = pd.DataFrame({"Document Name": st.session_state.document_names})
    st.dataframe(df_docs, use_container_width=True)


