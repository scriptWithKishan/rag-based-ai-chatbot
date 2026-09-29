from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import time
from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PDF_PATH, PINECONE_INDEX_NAME

# models/gemini-embedding-001 produces 3072-dimensional vectors
EMBEDDING_DIMENSION = 3072

# Free tier allows 100 embed requests/min; use a safe batch size with a cooldown
BATCH_SIZE = 80
BATCH_COOLDOWN_SECONDS = 65

def ensure_pinecone_index(index_name: str):
    """Create the Pinecone index if it doesn't already exist."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Index '{index_name}' not found. Creating it...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(3)
        print(f"Index '{index_name}' created and ready.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

def run_ingestion(pdf_path: str = PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    # 0. Ensure the Pinecone index exists
    ensure_pinecone_index(index_name)

    # 1. Load document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    # 2. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create embeddings & upload to Pinecone in rate-limited batches
    print(f"Uploading in batches of {BATCH_SIZE} (free tier: 100 req/min)...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}: embedding {len(batch)} chunks...")

        if vector_store is None:
            # First batch: create the vector store
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=index_name
            )
        else:
            # Subsequent batches: add to existing store
            vector_store.add_documents(batch)

        # Cooldown between batches to avoid hitting the free-tier rate limit
        if i + BATCH_SIZE < len(chunks):
            print(f"  Rate-limit cooldown: waiting {BATCH_COOLDOWN_SECONDS}s before next batch...")
            time.sleep(BATCH_COOLDOWN_SECONDS)

    print("All batches uploaded successfully!")
    return vector_store

if __name__ == "__main__":
    print(f"Starting ingestion for {PDF_PATH} into index '{PINECONE_INDEX_NAME}'...")
    store = run_ingestion()
    print("Ingestion complete successfully!")
