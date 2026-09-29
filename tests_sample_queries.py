import os
import sys
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

def run_tests():
    index_name = os.getenv("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
    print(f"Initializing LangGraph RAG pipeline using Pinecone index: '{index_name}'...\n")
    graph = build_rag_graph(index_name=index_name)

    sample_queries = [
        "What is an AI Agent and how does it differ from a standard LLM?",
        "What are the core components of an Agentic AI system architecture?",
        "How does tool calling work in agentic workflows?",
        "What is Retrieval-Augmented Generation (RAG) and why is it useful?",
        "What are the main evaluation metrics for RAG applications?",
        "What is the role of memory in agentic AI framework designs?"
    ]

    for idx, query in enumerate(sample_queries, 1):
        print("=" * 80)
        print(f"Query #{idx}: {query}")
        print("-" * 80)
        initial_state = {
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0
        }
        result = graph.invoke(initial_state)

        print(f"Final Answer:\n{result['answer']}\n")
        print(f"Confidence Score: {result['score']}\n")
        print("Retrieved Context Chunks:")
        for c_idx, chunk in enumerate(result['context'], 1):
            snippet = chunk[:200].replace('\n', ' ')
            print(f"  [{c_idx}] {snippet}...")
        print("=" * 80)
        print("\n")

if __name__ == "__main__":
    run_tests()
