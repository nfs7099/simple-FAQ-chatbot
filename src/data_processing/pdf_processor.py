import os
from pathlib import Path
from typing import List, Dict, Any, Generator, Iterator
import logging

import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    
    def __init__(self, pdf_dir: str):


        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_dir} does not exists, creating it...")
            self.pdf_dir.mkdir(parents=True, exist_ok=True)

    
    def get_pdf_files(self) -> List[Path]:
        logger.info(f"Getting pdf file list from {self.pdf_dir} ...")
        return list(self.pdf_dir.glob("*.pdf*"))
        
    
    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50)-> Iterator[str]:
        """Generator that yields chunks of text, breaking at sentence boundaries where possible"""
        if not text:
            return

        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate the end position
            end = min(start + chunk_size, text_length)

            # If we're not at the end of the text, try to find a sentence boundary
            if end < text_length:
                # Look for period, question mark, or exclamation point followed by space or newline
                for i in range(end - 1, max(start + chunk_size // 2, start), -1):
                    if text[i] in ['.', '!', '?'] and (i + 1 >= text_length or text[i + 1] in [' ', '\n']):
                        end = i + 1
                        break

            # Yield the chunk
            yield text[start:end].strip()

            # Move the starting position, accounting for overlap
            start = end - chunk_overlap

            if start >= text_length:
                break
        
    def process_pdf(self, pdf_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> Generator[Dict[str, Any], None, None]:
        """Process PDF one page at a time, yielding chunks"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    # Process one page at a time
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    
                    # Yield chunks for this page
                    for i, chunk in enumerate(self.chunk_text(page_text, chunk_size, chunk_overlap)):
                        yield {
                            "text": chunk,
                            "metadata": {
                                "source": pdf_path.name,
                                "page": page_num,
                                "chunk_id": i
                            }
                        }
                    # Clear page from memory
                    del page
                    del page_text
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

    def process_all_pdfs(self, chunk_size: int = 500, chunk_overlap: int = 50) -> Generator[Dict[str, Any], None, None]:
        """Process all PDFs, yielding chunks one at a time"""
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return

        for pdf_file in pdf_files:
            yield from self.process_pdf(pdf_file, chunk_size, chunk_overlap)


if __name__ == "__main__":

    # Run the PDF processor with sample configuration for testing
    from configs.config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    processor = PDFProcessor(PDF_DIR)
    chunk_count = 0
    sample_limit = 10  # Number of sample chunks to print

    for doc in processor.process_all_pdfs(CHUNK_SIZE, CHUNK_OVERLAP):
        chunk_count += 1
        if chunk_count <= sample_limit:
            print("Sample chunk:")
            print(f"Text: {doc['text']}")
            print(f"Metadata: {doc['metadata']}")
        else:
            break
    print(f"Processed {chunk_count} chunks from PDF files")





