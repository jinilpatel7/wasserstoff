import os
import shutil
import tempfile
import streamlit as st
from typing import List


UPLOAD_DIR = "AiInternTask/backend/data/uploads"

def save_uploaded_files(uploaded_files: List) -> List[str]:
    """Saves uploaded files to the local upload directory and returns paths."""
    saved_paths = []

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        final_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        shutil.move(temp_path, final_path)
        saved_paths.append(final_path)

    return saved_paths

