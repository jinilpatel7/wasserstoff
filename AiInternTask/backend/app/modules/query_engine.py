# This is my query_engine.py code 
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from .embedder import Embedder

load_dotenv()



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in environment variables")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

class QueryEngine:
    def __init__(self):
        self.embedder = Embedder()
        self.embedder.load_vectorstore()

        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
            temperature=0.3,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Use the following context from documents to answer the question.
            Respond with citations in the form "source: filename".

            Question: {question}
            ============  
            Context: {context}
            ============  
            Answer:
            """
        )
        self.output_parser = StrOutputParser()

        self.single_doc_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert assistant. You have the full text of one document below.
            Answer the question ONLY using this document's text.

            Provide a concise, accurate answer and include precise citations specifying page number, paragraph, or sentence if possible.

            Document text:
            {document_text}

            Question:
            {question}

            Answer (include citations with page/paragraph/sentence references):
            """
        )

    def answer_query(self, query: str) -> Dict[str, any]:
        docs = self.embedder.query(query, k=5)
        context = "\n\n".join([f"{doc.metadata['source']}:\n{doc.page_content}" for doc in docs])
        prompt_input = self.prompt.invoke({"question": query, "context": context})
        response = self.llm.invoke(prompt_input.to_messages())
        parsed_answer = self.output_parser.invoke(response)

        citations = [{"source": doc.metadata["source"], "snippet": doc.page_content[:300]} for doc in docs]
        return {"answer": parsed_answer, "citations": citations}

    def answer_query_single_document(self, query: str, document: Document) -> str:
        prompt_input = self.single_doc_prompt.invoke({
            "document_text": document.page_content,
            "question": query
        })
        response = self.llm.invoke(prompt_input.to_messages())
        return response.content

    def identify_themes(self, documents: List[Document]) -> List[Dict[str, any]]:
        if not documents:
            return []

        chunks = [f"{doc.metadata['source']}:\n{doc.page_content}" for doc in documents]
        joined_text = "\n\n".join(chunks[:10])  # avoid context length overflow

        theme_prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following document excerpts and identify key recurring themes or topics.
            For each theme, provide a brief summary and mention which documents support it.

            ============  
            {joined_text}
            ============  
            Output format:
            [
              {{
                "theme": "...",
                "summary": "...",
                "supporting_docs": ["filename1", "filename2"]
              }},
              ...
            ]
            """
        )
        prompt_input = theme_prompt.invoke({"joined_text": joined_text})
        response = self.llm.invoke(prompt_input.to_messages())

        try:
            import ast
            return ast.literal_eval(response.content.strip())
        except Exception:
            return []

