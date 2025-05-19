from typing import List, Dict
from langchain_core.documents import Document
from .embedder import Embedder
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import json

class QueryEngine:
    def __init__(self, model_name: str = "mistral"):
        self.embedder = Embedder()
        self.llm = Ollama(model=model_name)

        # Prompt for answering user queries
        answer_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                """
                You are an expert assistant helping analyze documents. 
                Given the following context, answer the user's question in a clear, concise, and well-formatted manner. 
                Use bullet points or numbered lists where appropriate.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=answer_prompt)

    def get_similar_docs(self, query: str, top_k: int = 5) -> List[Document]:
        return self.embedder.query(query, k=top_k)

    def get_qa_response(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])
        return self.chain.run({"context": context, "question": query})

    def extract_sources(self, docs: List[Document]) -> List[Dict]:
        citations = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            citations.append({"source": source, "snippet": snippet})
        return citations

    def answer_query(self, query: str) -> Dict:
        docs = self.get_similar_docs(query)
        answer = self.get_qa_response(query, docs)
        citations = self.extract_sources(docs)
        return {"answer": answer, "citations": citations}

    def identify_themes(self, docs: List[Document]) -> List[Dict]:
        if not docs:
            return []

        # Use smaller chunks and remove complex JSON prompt
        context = "\n\n".join([doc.page_content[:500] for doc in docs])
        prompt = (
            "Identify 3 to 5 key themes or topics discussed in the text below. "
            "For each theme, briefly explain it and mention which document (if possible). "
            "Keep the format simple and readable.\n\n"
            f"{context}\n\n"
            "Themes:"
        )

        response = self.llm(prompt).strip()

        # Return as a single block for display; we skip JSON formatting
        return [{"theme": "Identified Themes", "summary": response, "supporting_docs": []}]

