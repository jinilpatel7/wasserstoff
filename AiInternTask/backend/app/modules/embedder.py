import os
from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./AiInternTask/backend/data/chroma_db")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


class Embedder:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)
        self.db_path = CHROMA_DB_PATH
        self.vector_store = None

    def _prepare_documents(self, text_dict: Dict[str, str]) -> List[Document]:
        docs = []
        for source, text in text_dict.items():
            docs.append(Document(page_content=text, metadata={"source": source}))
        return docs

    def store_embeddings(self, text_dict: Dict[str, str]):
        documents = self._prepare_documents(text_dict)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )

    def load_vectorstore(self):
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    def query(self, user_query: str, k: int = 5) -> List[Document]:
        if self.vector_store is None:
            self.load_vectorstore()
        return self.vector_store.similarity_search(user_query, k=k)

