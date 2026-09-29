from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME
import os

app = FastAPI(title="Agentic AI RAG API")

graph_instance = None

def get_graph():
    global graph_instance
    if graph_instance is None:
        index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
        graph_instance = build_rag_graph(index_name=index_name)
    return graph_instance

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    rag_graph = get_graph()
    initial_state = {"question": request.query, "context": [], "answer": "", "score": 0.0}
    result = rag_graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"]
    )

# Streamlit Interface Integration
try:
    import streamlit as st

    # Only render Streamlit elements if executing in a Streamlit context
    if st.runtime.exists():
        st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")
        st.title("🤖 Agentic AI RAG Chatbot")
        st.write("Ask questions based on the uploaded Ebook-Agentic-AI document.")

        query = st.text_input("Enter your question:")
        if st.button("Submit Query") and query.strip():
            with st.spinner("Retrieving context and generating answer..."):
                rag_graph = get_graph()
                state = {"question": query, "context": [], "answer": "", "score": 0.0}
                res = rag_graph.invoke(state)

                st.markdown("### Answer")
                st.write(res["answer"])

                st.markdown(f"**Confidence Score:** {res['score']}")

                with st.expander("Retrieved Context Chunks"):
                    for idx, chunk in enumerate(res["context"], 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.text(chunk)
except Exception:
    pass
