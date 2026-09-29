import streamlit as st
from dotenv import load_dotenv
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME
import os

load_dotenv()

st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI RAG Chatbot")
st.write("Ask questions based on the uploaded Ebook-Agentic-AI document.")

@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_graph():
    index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
    return build_rag_graph(index_name=index_name)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "score" in message:
            st.caption(f"Confidence Score: {message['score']:.2f}")

# Chat input at the bottom
if query := st.chat_input("Ask a question about the document..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                rag_graph = get_graph()
                state = {"question": query, "context": [], "answer": "", "score": 0.0}
                res = rag_graph.invoke(state)

                answer = res["answer"]
                score = res["score"]
                context_chunks = res["context"]

                st.markdown(answer)
                st.caption(f"Confidence Score: {score:.2f}")

                with st.expander("📄 Retrieved Context Chunks"):
                    for idx, chunk in enumerate(context_chunks, 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.text(chunk)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "score": score
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
