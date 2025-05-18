from typing import List, Dict
from langchain_core.documents import Document
from .embedder import Embedder
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama

class QueryEngine:
    def __init__(self, model_name: str = "mistral"):
        self.embedder = Embedder()
        self.llm = Ollama(model=model_name)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def get_similar_docs(self, query: str, top_k: int = 5) -> List[Document]:
        """Search Chroma vector store for top K similar documents."""
        return self.embedder.query(query, k=top_k)

    def get_qa_response(self, query: str, docs: List[Document]) -> str:
        """Generate answer from relevant documents using local LLM."""
        result = self.chain.run(input_documents=docs, question=query)
        return result

    def extract_sources(self, docs: List[Document]) -> List[Dict]:
        """Extract source citations from matched documents."""
        citations = []
        for doc in docs:
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        return citations

    def answer_query(self, query: str) -> Dict:
        """Full pipeline: retrieve documents, synthesize response, return citations."""
        docs = self.get_similar_docs(query)
        answer = self.get_qa_response(query, docs)
        citations = self.extract_sources(docs)

        return {
            "answer": answer,
            "citations": citations
        }
