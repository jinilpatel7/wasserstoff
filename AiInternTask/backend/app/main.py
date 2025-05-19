import os
import sys
import streamlit as st
import pandas as pd

# Extend import path to include 'modules'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from modules.upload import save_uploaded_files
from modules.text_extractor import extract_all_text
from modules.ocr_processor import process_images
from modules.embedder import Embedder
from modules.query_engine import QueryEngine

st.set_page_config(page_title="AI Document Chatbot", layout="wide")
st.title("📚 AI Document Research & Theme Identification Chatbot")

DOCS_FOLDER = "AiInternTask/backend/data/uploads"

embedder = Embedder()
query_engine = QueryEngine()

if "extracted_docs" not in st.session_state:
    st.session_state.extracted_docs = {}

# Upload documents
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or images",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Saving and processing uploaded documents..."):
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        saved_paths = save_uploaded_files(uploaded_files)

        pdf_files = [f for f in saved_paths if f.lower().endswith(".pdf")]
        image_files = [f for f in saved_paths if not f.lower().endswith(".pdf")]

        if not st.session_state.extracted_docs:
            ocr_texts = process_images(image_files) if image_files else {}
            pdf_texts = extract_all_text(pdf_files) if pdf_files else {}
            st.session_state.extracted_docs.update({**pdf_texts, **ocr_texts})

            embedder.store_embeddings(st.session_state.extracted_docs)
            st.success(f"Processed {len(saved_paths)} document(s) successfully!")

# Query section
if st.session_state.extracted_docs:
    st.sidebar.header("2. Ask a Question")
    query = st.sidebar.text_area(
        "Enter your query",
        placeholder="e.g. What are the deliverables in the internship task?"
    )

    if st.sidebar.button("Get Answer") and query.strip():
        with st.spinner("Processing your query..."):
            result = query_engine.answer_query(query)

            # 📝 Answer
            st.markdown("### 📝 Answer")
            st.markdown(result["answer"], unsafe_allow_html=True)

            # 📑 Citations
            if result["citations"]:
                st.markdown("### 📑 Source Citations")
                df = pd.DataFrame(result["citations"])
                st.dataframe(df, use_container_width=True)

            # 🎯 Themes
            # 🎯 Themes
            st.markdown("### 🎯 Identified Themes")
            docs_for_themes = query_engine.get_similar_docs(query)
            themes = query_engine.identify_themes(docs_for_themes)

            if themes:
                for theme in themes:
                    st.markdown(f"**🟢 {theme['theme']}**")
                    st.write(theme["summary"])
            else:
                st.info("No themes identified.")

else:
    st.info("Upload some documents first to start querying.")
