import glob
import logging
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import MarkdownRenderer
from marker.util import classes_to_strings
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run():
    convert_dir = "./docs-to-convert"
    target_dir = "./docs"

    pdf_files = [os.path.join(convert_dir, fn) for fn in next(os.walk(convert_dir))[2]]

    for pdf_file in pdf_files:
        file = pdf_file
        if not isinstance(pdf_file, Path):
            file = Path(pdf_file)
        logger.info(f"Converting {file}")

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            renderer=classes_to_strings([MyMarkdownRenderer])[0]
        )

        rendered = converter(file.__str__())
        text, _, images = text_from_rendered(rendered)

        file_name_md = file.name.rsplit(".", 1)[0] + ".md"
        with open(f"{target_dir}/{file_name_md}", "wb") as f:
            f.write(text.encode("utf-8"))

class MyMarkdownRenderer(MarkdownRenderer):
    def __init__(self) -> None:
        logger.info("Created MyMarkdownRenderer")
        super().__init__()
        self.paginate_output = True

if __name__ == "__main__":
    run()
