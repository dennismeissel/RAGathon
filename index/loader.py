import json
import os
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List

from fsspec import AbstractFileSystem
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader
from tenacity import stop_after_attempt, retry

logger = logging.getLogger(__name__)
RETRY_TIMES = 1


def load_documents():
    """Returns all documents at once. Useful, for running one ingestion pipeline with all docs.
    Will not create DB if error during processing or embedding for any doc."""
    reader = get_dir_reader()
    return reader.load_data()


def load_documents_with_iter():
    """Returns an iterator with all documents. Useful, for running the full ingestion pipeline for every doc."""
    reader = get_dir_reader()
    return reader.iter_data()

def get_dir_reader():
    custom_file_extractor = {
        ".md": MarkerMDReader()
    }
    docs_dir = os.getenv("DOC_INGESTION_DIR", "./docs")
    logger.info(f"input_dir [{docs_dir}]")
    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        recursive=True,
        filename_as_id=True,
        raise_on_error=True,

        file_extractor=custom_file_extractor,
    )
    return reader


class MarkerMDReader(BaseReader):
    """Marker PDF Reader that reads md files converted beforehand.
    Returns a list of Documents with each doc representing the converted Markdown representing the content of one page in the original document.
    This is a Reader implementation compatible with Llama-Index ingestion pipelines"""

    def __init__(self) -> None:
        """
        Initialize MarkerMDReader.
        """
        logger.info("Init MarkerMDReader")

    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    def load_data(
            self,
            file: Path,
            extra_info: Optional[Dict] = None,
            fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        logger.info(f"Loading file {file}")
        if not isinstance(file, Path):
            file = Path(file)

        doc_pages: List[Document] = []

        subset_meta_data = load_subset_metadata(file)
        logger.info(f"subset_meta_data [{subset_meta_data}]")

        with open(file, "r", encoding="utf-8") as md_file:
            md_content = md_file.read()
            pages = re.split(r"^.(\d+).-{48}\n\n", md_content,
                             flags=re.MULTILINE)  # includes capture group with page index in string
            # regex returns: [0, text, 1, text, 2, text, ...]
            del pages[0]  # clear first item as it is an empty string before the first page separator created by the regex
            # Iterate over the array in pairs
            for i in range(0, len(pages), 2):
                page_index = int(pages[i])
                page_text = pages[i + 1]
                metadata = {**{"file_name": file.name, "page": page_index}, **subset_meta_data}
                doc_pages.append(Document(text=page_text, metadata=metadata))

        return doc_pages


def load_subset_metadata(file: Path) -> Dict:
    logger.debug("Start Loading subset.json metadata")
    add_subset_metadata = os.getenv("ADD_SUBSET_METADATA", "False").lower() == "true"

    base = os.path.basename(file)
    file_name_without_extension, _ = os.path.splitext(base)

    if add_subset_metadata:
        try:
            with open("./docs-subset/subset.json", 'r') as file:
                data = json.load(file)

                for obj in data:
                    if obj.get('sha1') == file_name_without_extension:
                        return obj
                logger.debug("Done Loading subset.json metadata")
                return {}

        except FileNotFoundError:
            logger.error(f"The file {file_name_without_extension} does not exist.")
        except json.JSONDecodeError:
            logger.error(f"The file {file_name_without_extension} is not a valid JSON file.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
    return {}
