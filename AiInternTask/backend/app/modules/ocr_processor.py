import os
from PIL import Image
import pytesseract
from typing import Dict, List

SUPPORTED_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]

def is_image_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in SUPPORTED_IMAGE_EXTS

def extract_text_from_image(file_path: str) -> str:
    """Performs OCR on a single image file."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR for {file_path}: {e}")
        return ""

def process_images(file_paths: List[str]) -> Dict[str, str]:
    """Processes all image files and returns extracted text per file."""
    extracted_texts = {}
    for file_path in file_paths:
        if is_image_file(file_path):
            extracted_texts[file_path] = extract_text_from_image(file_path)
    return extracted_texts
