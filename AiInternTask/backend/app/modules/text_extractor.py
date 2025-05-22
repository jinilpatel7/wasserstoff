import os
import fitz  # PyMuPDF
from typing import Dict, List
from .ocr_processor import is_image_file, process_images
import docx

SUPPORTED_PDF_EXTS = [".pdf"]
SUPPORTED_TEXT_EXTS = [".txt"]
SUPPORTED_WORD_EXTS = [".docx"]

def is_pdf_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in SUPPORTED_PDF_EXTS

def is_txt_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in SUPPORTED_TEXT_EXTS

def is_docx_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in SUPPORTED_WORD_EXTS

def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        return "".join(page.get_text() for page in doc).strip()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def extract_all_text(file_paths: List[str]) -> Dict[str, str]:
    extracted_data = {}

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()

        if is_image_file(file_path):
            extracted_data.update(process_images([file_path]))
        elif is_pdf_file(file_path):
            extracted_data[file_path] = extract_text_from_pdf(file_path)
        elif is_txt_file(file_path):
            extracted_data[file_path] = extract_text_from_txt(file_path)
        elif is_docx_file(file_path):
            extracted_data[file_path] = extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file type: {file_path}")

    return extracted_data
