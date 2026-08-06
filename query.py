import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "codebase"


def load_index():
    """Load the already-built index from disk — no re-cloning, no re-embedding."""

    api_key = os.getenv("GOOGLE_API_KEY")

    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.llm = GoogleGenAI(
        model_name="models/gemini-2.0-flash",
        api_key=api_key,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR):
        print("No index found. Run ingest.py first to build one.")
        exit()

    print("Loading index...")
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=8)
    print("Ready! Ask questions about the codebase (type 'exit' to quit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\nAssistant: {response}\n")