import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index-gemini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = "models/gemini-embedding-001"  # 3072 dims, free via Gemini API
LLM_MODEL = "gemini-2.5-flash"                   # Free via Gemini API
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_PATH = os.path.join("data", "Ebook-Agentic-AI.pdf")
