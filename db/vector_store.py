import os

from llama_index.vector_stores.chroma import ChromaVectorStore


def get_vector_store():
    """Returns the chroma db vector store as llama index vector store"""
    collection_name = os.getenv("CHROMA_COLLECTION", "default")
    chroma_path = os.getenv("CHROMA_PATH", "./db/chroma-db")
    return ChromaVectorStore.from_params(
        persist_dir=chroma_path, collection_name=collection_name
    )
