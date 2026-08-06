import os
import stat
import shutil
from git import Repo
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb


load_dotenv()

CLONE_DIR = "cloned_repo"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "codebase"


def _remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def clone_repo(repo_url: str) -> str:
    
    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR, onerror=_remove_readonly)

    print(f"Cloning {repo_url} ...")
    Repo.clone_from(repo_url, CLONE_DIR)
    print("Clone complete.")
    return CLONE_DIR


def build_index(source_dir: str):
    """Read all code files, chunk them, embed them, and store in ChromaDB."""

    
    api_key = os.getenv("GOOGLE_API_KEY")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    Settings.llm = GoogleGenAI(
        model_name="models/gemini-2.0-flash",
        api_key=api_key,
    )

    print("Reading files from cloned repo...")
    
    documents = SimpleDirectoryReader(
        input_dir=source_dir,
        recursive=True,
        required_exts=[
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
            ".rb", ".cpp", ".c", ".h", ".md", ".json", ".yaml", ".yml",
        ],
    ).load_data()
    print(f"Loaded {len(documents)} files.")


    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building index (this embeds every chunk — may take a bit)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print("Index built and saved to disk.")
    return index


if __name__ == "__main__":
    repo_url = input("Enter a GitHub repo URL to ingest: ").strip()
    clone_repo(repo_url)
    build_index(CLONE_DIR)
    print("\nDone! You can now run query.py to ask questions about this codebase.")