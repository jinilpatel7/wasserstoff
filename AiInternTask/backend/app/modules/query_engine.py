from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import OllamaLLM
from .embedder import Embedder


class QueryEngine:
    def __init__(self, model_name: str = "mistral"):
        self.embedder = Embedder()
        self.llm = OllamaLLM(model=model_name)

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

        self.chain: RunnableSequence = answer_prompt | self.llm

    def get_similar_docs(self, query: str, top_k: int = 5) -> List[Document]:
        return self.embedder.query(query, k=top_k)

    def get_qa_response(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])
        return self.chain.invoke({"context": context, "question": query})

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

        grouped_docs = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            grouped_docs.setdefault(source, []).append(doc.page_content)

        condensed_contexts = []
        for source, texts in grouped_docs.items():
            combined_text = "\n".join(texts)
            condensed_contexts.append(f"[{source}]:\n{combined_text[:1000]}")  # Limit per document

        full_context = "\n\n".join(condensed_contexts)

        theme_prompt = (
            "You are an expert analyzing multiple documents to extract recurring themes.\n"
            "Below are excerpts from several documents. Identify 3 to 5 key themes that appear across them.\n"
            "For each theme:\n"
            "- Name the theme\n"
            "- Briefly explain it\n"
            "- List the document sources where this theme appears (like [doc1.pdf])\n\n"
            "Themes:\n\n"
            f"{full_context}"
        )

        try:
            response = self.llm.invoke(theme_prompt).strip()
            return [{"theme": "Identified Themes", "summary": response, "supporting_docs": list(grouped_docs.keys())}]
        except Exception as e:
            print(f"Error identifying themes: {e}")
            return []

