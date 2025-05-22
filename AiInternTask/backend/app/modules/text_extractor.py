import os
import fitz  # PyMuPDF
from typing import Dict, List
from .ocr_processor import is_image_file, process_images

SUPPORTED_PDF_EXTS = [".pdf"]

def is_pdf_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in SUPPORTED_PDF_EXTS

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_all_text(file_paths: List[str]) -> Dict[str, str]:
    """Extracts text from all uploaded files (PDF or images)."""
    extracted_data = {}

    image_files = [fp for fp in file_paths if is_image_file(fp)]
    pdf_files = [fp for fp in file_paths if is_pdf_file(fp)]

    # Process images via OCR
    if image_files:
        extracted_data.update(process_images(image_files))

    # Process PDFs
    for file_path in pdf_files:
        text = extract_text_from_pdf(file_path)
        extracted_data[file_path] = text

    return extracted_data

