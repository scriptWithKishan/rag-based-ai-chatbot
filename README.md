# RAG-based AI Chatbot 🤖

A modular, production-ready Retrieval-Augmented Generation (RAG) AI Chatbot built using **LangGraph**, **LangChain**, **Google Gemini** (`gemini-embedding-001` and `gemini-2.5-flash`), **Pinecone**, **FastAPI**, and **Streamlit**.

---

## 📁 Project Structure

```text
rag-based-ai-chatbot/
│
├── data/
│   └── Ebook-Agentic-AI.pdf        # Source document for ingestion
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py                # PDF loading, chunking & Pinecone index setup
│   ├── graph.py                    # LangGraph workflow definition & state logic
│   └── config.py                   # Environment setup & constants
│
├── app.py                          # FastAPI application entry point
├── streamlit_app.py                # Streamlit chat UI
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for environment variables
├── README.md                       # Deployment & usage instructions
└── tests_sample_queries.py         # Script with 6 sample grounding test queries
```

---


### Components Summary

- **Ingestion Pipeline (`src/ingestion.py`)**: Loads `Ebook-Agentic-AI.pdf`, splits text using `RecursiveCharacterTextSplitter` into chunks of 1000 characters with 200-character overlap, embeds them using `gemini-embedding-001` (3072 dims), and uploads vector representations to Pinecone in **rate-limited batches of 80** to respect the free-tier quota (100 req/min).
- **LangGraph RAG Workflow (`src/graph.py`)**: Manages state transitions across `AgentState` (`question`, `context`, `answer`, `score`), executing a `retrieve` node followed by a strict prompt-grounded `generate` node using `gemini-2.5-flash`.
- **Streamlit UI (`streamlit_app.py`)**: Interactive chat interface with persistent message history, confidence score display, and expandable context chunk viewer.
- **FastAPI Layer (`app.py`)**: Exposes `POST /chat` returning a JSON payload with `final_answer`, `retrieved_context`, and `confidence_score`.
- **Test Suite (`tests_sample_queries.py`)**: Runs 6 document-grounded evaluation queries verifying retrieval precision and confidence scoring.

---

## ⚙️ Environment Setup

### 1. Clone the repository & create virtual environment

```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 2. Install project dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

`.env` content:

```env
OPENAI_API_KEY=your_openai_api_key        # Optional (not used currently)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index-gemini
GOOGLE_API_KEY=your_gemini_api_key        # Get free key at aistudio.google.com/apikey
```

> **Note:** Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Get a free Pinecone API key at [pinecone.io](https://www.pinecone.io/).

---

## 🚀 Execution & Usage

### 1. Run Data Ingestion

Populate the Pinecone index with vectors generated from `data/Ebook-Agentic-AI.pdf`.  
This step auto-creates the index if it doesn't exist and uploads embeddings in rate-limited batches:

```bash
python -m src.ingestion
```


---

### 2. Launch Streamlit Chat UI

Run the interactive chatbot:

```bash
python -m streamlit run streamlit_app.py
```

Open in your browser: [http://localhost:8501](http://localhost:8501)

Features:
- Persistent chat history across messages
- Confidence score per answer
- Expandable retrieved context chunks

---

### 3. Launch FastAPI Server (Optional)

Start the REST API server:

```bash
python -m uvicorn app:app --reload --port 8000
```

- Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Example `POST /chat` request:
  ```json
  { "query": "What is an AI Agent?" }
  ```
- Example response:
  ```json
  {
    "final_answer": "An AI Agent is an autonomous entity that perceives its environment...",
    "retrieved_context": ["Chunk text content..."],
    "confidence_score": 0.95
  }
  ```

---

## 🧪 Running Tests

The test suite runs **6 document-grounded queries** through the full RAG pipeline and prints the answers, confidence scores, and retrieved context snippets.

### Run all sample queries

```bash
python tests_sample_queries.py
```

### What the tests cover

| # | Query |
|---|-------|
| 1 | What is an AI Agent and how does it differ from a standard LLM? |
| 2 | What are the core components of an Agentic AI system architecture? |
| 3 | How does tool calling work in agentic workflows? |
| 4 | What is Retrieval-Augmented Generation (RAG) and why is it useful? |
| 5 | What are the main evaluation metrics for RAG applications? |
| 6 | What is the role of memory in agentic AI framework designs? |

### Expected output format

```
================================================================================
Query #1: What is an AI Agent and how does it differ from a standard LLM?
--------------------------------------------------------------------------------
Final Answer:
An AI Agent is an autonomous system that...

Confidence Score: 0.95

Retrieved Context Chunks:
  [1] AI agents are systems that perceive their environment and take actions...
  [2] Unlike standard LLMs which are stateless, agents maintain context...
================================================================================
```

> **Prerequisites:** Make sure you have run `python -m src.ingestion` before running tests so the Pinecone index is populated.

---

## 🔑 API Keys & Free Tier Limits

| Service | Free Tier Limit | Get Key |
|---|---|---|
| **Google Gemini** | 100 embed req/min, 15 chat req/min | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Pinecone** | 2 free indexes, 100K vectors | [pinecone.io](https://www.pinecone.io/) |
