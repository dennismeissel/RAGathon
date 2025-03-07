import os

from llama_index.core import Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding


def init_settings():
    """Initializes llama index settings."""
    Settings.chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "8192"))
    config = {
        "api_key": os.getenv("EMBEDDING_API_KEY"),
        "base_url": os.getenv("EMBEDDING_BASE_URL"),
        "model": os.getenv("EMBEDDING_MODEL"),
        "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    }
    Settings.embed_model = NVIDIAEmbedding(**config)
