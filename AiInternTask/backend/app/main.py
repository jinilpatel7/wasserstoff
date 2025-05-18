import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.text_extractor import extract_all_text
from modules.ocr_processor import process_images
from modules.embedder import Embedder
from modules.query_engine import QueryEngine

st.set_page_config(page_title="AI Document Chatbot", layout="wide")
st.title("📚 AI Document Research & Theme Identification Chatbot")

DOCS_FOLDER = "C:/Users/jinil/Desktop/wasserstoff/AiInternTask/demo"
EXTRACTION_FOLDER = "C:/Users/jinil/Desktop/wasserstoff/AiInternTask/backend/data"

embedder = Embedder()
query_engine = QueryEngine()

st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or images", type=["pdf", "png", "jpg", "jpeg", "tiff"], accept_multiple_files=True
)

extracted_docs = None

if uploaded_files:
    with st.spinner("Saving and processing uploaded documents..."):
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        saved_paths = []

        for file in uploaded_files:
            save_path = os.path.join(DOCS_FOLDER, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(save_path)

        pdf_files = [f for f in saved_paths if f.lower().endswith(".pdf")]
        image_files = [f for f in saved_paths if not f.lower().endswith(".pdf")]

        ocr_texts = {}
        if image_files:
            st.info("Running OCR on image files...")
            ocr_texts = process_images(image_files)  # Returns dict {filename: text}

        # Extract text from PDFs
        pdf_texts = extract_all_text(pdf_files)  # ONLY pass pdf_files list

        # Combine extracted docs: OCR texts + PDF texts
        # Assume both pdf_texts and ocr_texts are dicts: {filename: text}
        extracted_docs = {**pdf_texts, **ocr_texts}

        # Add documents to embedder
        embedder.store_embeddings(extracted_docs)


        st.success(f"Successfully processed {len(saved_paths)} document(s)!")

# Show query box **only if documents uploaded and processed**
if extracted_docs:
    st.sidebar.header("2. Ask a Question")
    query = st.sidebar.text_area(
        "Enter your query", placeholder="e.g. What is the penalty under SEBI Act?", height=100
    )

    if st.sidebar.button("Submit Query") and query.strip():
        with st.spinner("Processing your query..."):
            result = query_engine.answer_query(query)

            st.subheader("💬 Synthesized Answer")
            st.write(result["answer"])

            st.subheader("📑 Source Citations")
            for i, citation in enumerate(result["citations"], 1):
                st.markdown(f"**{i}. Source**: `{citation['source']}`")
                st.markdown(f"> {citation['snippet']}")

            st.success("Query answered with relevant sources.")
else:
    st.info("Please upload documents to begin.")
