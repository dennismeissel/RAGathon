import logging

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.vector_store import get_vector_store
from doc_store import get_doc_metadata_store, persist_doc_metadata_store
from loader import load_documents, load_documents_with_iter
from md_processor import MarkdownProcessor
from settings import init_settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(doc_store, vector_store, documents):
    preprocessor = MarkdownProcessor()
    md_splitter = MarkdownNodeParser()
    max_chunk_size_splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )
    transformations = [preprocessor, md_splitter, max_chunk_size_splitter, Settings.embed_model]
    pipeline = IngestionPipeline(
        transformations=transformations,
        docstore=doc_store,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        vector_store=vector_store,
    )
    nodes = pipeline.run(show_progress=True, documents=documents)
    return nodes

def run_pipeline_for_docs(documents):
    doc_metadata_store = get_doc_metadata_store()
    vector_store = get_vector_store()
    _ = run_pipeline(doc_metadata_store, vector_store, documents)
    persist_doc_metadata_store(doc_metadata_store, vector_store)

def run_doc_ingestion():
    logger.info("Run document ingestion pipeline")
    load_dotenv()
    init_settings()

    run_per_doc = os.getenv("RUN_PIPELINE_PER_DOC", "True").lower() == "true"
    if run_per_doc:
        logger.info("Running ingestion pipeline for each doc in dir")
        doc_iter = load_documents_with_iter()
        i = 1
        for docs in doc_iter:
            logger.info(f"Indexing doc {i}")
            run_pipeline_for_docs(docs)
            i = i + 1

    else:
        logger.info("Running ingestion pipeline for all docs in dir")
        documents = load_documents()
        run_pipeline_for_docs(documents)

    logger.info("Finished generating the index")


if __name__ == "__main__":
    run_doc_ingestion()
