import logging
import re
from typing import Sequence, Any, List

from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

logger = logging.getLogger(__name__)

class DocPage:
    """Mutable class for documents"""
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


class MarkdownProcessor(NodeParser):
    """Processor for Markdown. To be used before actual Markdown Chunking.

    Moves all content that was wrapped to a new page but belongs to the heading in the current page to the current page.

    Usefully, if PDFs are parsed per page.
    """

    def _parse_nodes(
            self,
            nodes: Sequence[BaseNode],
            show_progress: bool = False,
            **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes."""
        all_nodes: List[BaseNode] = []

        doc_pages: List[DocPage] = list(map(self._convert_from_document, list(nodes)))
        nodes_with_progress = list(get_tqdm_iterable(doc_pages, show_progress, "Postprocessing Markdown nodes"))

        doc_pages: List[DocPage] = self._post_process_markdown(nodes_with_progress)

        all_nodes.extend(list(map(self._convert_to_document, doc_pages)))

        return all_nodes

    @staticmethod
    def _convert_to_document(doc_page: DocPage) -> Document:
        return Document(text=doc_page.text, metadata=doc_page.metadata)

    @staticmethod
    def _convert_from_document(node: BaseNode) -> DocPage:
        return DocPage(text=node.text, metadata=node.metadata)

    def _post_process_markdown(self, pages: List[DocPage]) -> List[DocPage]:
        """Post processes converter markdown pages:
        - moves all content that is wrapped to the next page the previous document
        """
        processed_pages: List[DocPage] = []
        prev_page = None
        for i, page in enumerate(pages):
            if i == 0:
                processed_pages.append(page)
                prev_page = page
                continue
            if not self._page_starts_with_heading(page):
                logger.debug(f"Post processing page index [{i}]")
                content_belonging_to_prev_heading = self._remove_content_before_next_heading(page)
                prev_page.text += "\n" + content_belonging_to_prev_heading
                current_page_nr = str(page.metadata["page"])
                try:
                    prev_page.metadata["additional_pages"] += f";{current_page_nr}"
                except KeyError:
                    prev_page.metadata["additional_pages"] = current_page_nr
            if page.text:
                processed_pages.append(page)  # only add page if it still has text
                prev_page = page

        return processed_pages

    @staticmethod
    def _page_starts_with_heading(page: DocPage) -> bool:
        first_non_whitespace_char_index = next((i for i, ch in enumerate(page.text) if not ch.isspace()), -1)
        if first_non_whitespace_char_index == -1:
            return False
        first_char = page.text[first_non_whitespace_char_index] if first_non_whitespace_char_index < len(
            page.text) else None
        return first_char == "#"

    @staticmethod
    def _remove_content_before_next_heading(page: DocPage) -> str:
        """Removes the content of the page before the next Markdown heading and returns the removed content as a string"""
        markdown = page.text
        heading_re = re.compile(r'^(#{1,6}\s)', re.MULTILINE)

        headings = [match for match in heading_re.finditer(markdown)]

        # If there's no heading, return the whole markdown
        if not headings:
            # remove all content from current page as no heading present
            page.text = None
            return markdown.strip()

        first_heading_start_index = headings[0].start()

        # Content is everything before this heading
        content = markdown[:first_heading_start_index].strip()

        # remove the content from the current page
        page.text = page.text[first_heading_start_index:]

        return content
