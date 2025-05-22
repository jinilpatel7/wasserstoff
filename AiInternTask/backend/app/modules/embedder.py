import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./backend/data/chroma_db")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class Embedder:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=HF_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
            )
        self.db_path = CHROMA_DB_PATH
        self.vector_store = None

    def _prepare_documents(self, text_dict: Dict[str, str]) -> List[Document]:
        return [
            Document(page_content=text, metadata={"source": source})
            for source, text in text_dict.items()
        ]

    def load_vectorstore(self):
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    def store_embeddings(self, text_dict: Dict[str, str]):
        if self.vector_store is None:
            if os.path.exists(os.path.join(self.db_path, "chroma.sqlite3")):
                self.load_vectorstore()
            else:
                self.vector_store = Chroma.from_documents(
                    documents=[],
                    embedding=self.embedding_model,
                    persist_directory=self.db_path
                )

        existing_sources = set()
        if self.vector_store is not None:
            existing_docs = self.vector_store.get()
            existing_sources = set(meta['source'] for meta in existing_docs['metadatas'] if 'source' in meta)

        new_docs = []
        for source, text in text_dict.items():
            if source not in existing_sources:
                new_docs.append(Document(page_content=text, metadata={"source": source}))

        if new_docs:
            self.vector_store.add_documents(new_docs)

    def query(self, user_query: str, k: int = 5) -> List[Document]:
        if self.vector_store is None:
            self.load_vectorstore()
        return self.vector_store.similarity_search(user_query, k=k)


