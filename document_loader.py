import os
import shutil
from llama_index.core import SimpleDirectoryReader
import config

def save_uploaded_files(uploaded_files) -> list:
    """
    Saves uploaded files from Streamlit into the configured upload directory.
    Returns a list of absolute file paths to the saved files.
    """
    if not os.path.exists(config.UPLOAD_DIR):
        os.makedirs(config.UPLOAD_DIR)
        
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_documents_from_dir(dir_path: str = None) -> list:
    """
    Loads and parses PDF documents from the specified directory.
    If no directory is specified, config.UPLOAD_DIR is used.
    """
    if dir_path is None:
        dir_path = config.UPLOAD_DIR
        
    if not os.path.exists(dir_path) or not os.listdir(dir_path):
        return []
        
    # Use SimpleDirectoryReader to recursively parse PDFs
    loader = SimpleDirectoryReader(
        input_dir=dir_path,
        required_exts=[".pdf"],
        recursive=True
    )
    return loader.load_data()

def clear_upload_dir():
    """
    Clears all files in the configured upload directory.
    """
    if os.path.exists(config.UPLOAD_DIR):
        shutil.rmtree(config.UPLOAD_DIR)
    os.makedirs(config.UPLOAD_DIR)
