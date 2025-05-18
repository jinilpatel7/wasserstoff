import os
from typing import List, Dict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
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
        """Convert raw text into LangChain Document objects."""
        docs = []
        for source, text in text_dict.items():
            docs.append(Document(page_content=text, metadata={"source": source}))
        return docs

    def store_embeddings(self, text_dict: Dict[str, str]):
        """Generate and persist embeddings from the document dictionary."""
        documents = self._prepare_documents(text_dict)

        # Create and persist Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )
        self.vector_store.persist()

    # Add this method to support your current main.py call
    def add_documents(self, text_dict: Dict[str, str]):
        """Alias for store_embeddings to add documents to vector store."""
        self.store_embeddings(text_dict)

    def load_vectorstore(self):
        """Load existing vector store from disk."""
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    def query(self, user_query: str, k: int = 5) -> List[Document]:
        """Query the vector store using a user prompt."""
        if self.vector_store is None:
            self.load_vectorstore()
        return self.vector_store.similarity_search(user_query, k=k)
