import os

from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

STORE_METADATA_DIR = os.getenv("STORE_METADATA_DIR", "docs-store-metadata-dir")


def get_doc_metadata_store():
    """Returns the document metadata store. From existing folder containing the metadata for already ingested documents or a new doc store."""
    if os.path.exists(STORE_METADATA_DIR):
        return SimpleDocumentStore.from_persist_dir(STORE_METADATA_DIR)
    else:
        return SimpleDocumentStore()


def persist_doc_metadata_store(doc_store, vector_store):
    """Persists document metadata to store metadata dir."""
    storage_context = StorageContext.from_defaults(
        docstore=doc_store,
        vector_store=vector_store,
    )
    storage_context.persist(STORE_METADATA_DIR)
