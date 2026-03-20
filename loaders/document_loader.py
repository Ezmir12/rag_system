"""
Multimodal document loader: PDF, images (PNG, JPG, JPEG), and text files.
"""
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# Image extensions supported for OCR-based loading
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}


def _load_pdf(file_path: str) -> list:
    """Load PDF via PyPDFLoader."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        print("Warning: The PDF appears to be empty or image-only.")
    return documents


def _load_image_unstructured(file_path: str, filename: str) -> list:
    """Load image using UnstructuredImageLoader (requires unstructured package)."""
    try:
        from langchain_community.document_loaders import UnstructuredImageLoader
        loader = UnstructuredImageLoader(file_path)
        return loader.load()
    except ImportError:
        print(
            "Note: For image OCR, install: pip install 'unstructured[all-docs]' "
            f"Skipping text extraction for {filename}."
        )
        # Fallback: create a minimal document with image path in metadata
        return [
            Document(
                page_content=f"[Image file: {filename} - Install 'unstructured[all-docs]' for OCR]",
                metadata={"source": file_path, "type": "image", "filename": filename},
            )
        ]


def _load_image_pytesseract(file_path: str, filename: str) -> list:
    """Fallback: load image using pytesseract + PIL if available."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            text = f"[Image: {filename} - No text detected]"
        return [
            Document(
                page_content=text,
                metadata={"source": file_path, "type": "image", "filename": filename},
            )
        ]
    except ImportError:
        return _load_image_unstructured(file_path, filename)


def _load_text_file(file_path: str, filename: str) -> list:
    """Load plain text, markdown, or CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return [
        Document(
            page_content=content,
            metadata={"source": file_path, "type": "text", "filename": filename},
        )
    ]


def load_documents(file_path: str, filename: str = None) -> list:
    """
    Load documents from file. Supports:
    - PDF (.pdf)
    - Images (.png, .jpg, .jpeg, etc.) via UnstructuredImageLoader or pytesseract
    - Text (.txt, .md, .csv)
    """
    filename = filename or os.path.basename(file_path)
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in IMAGE_EXTENSIONS:
        # Try pytesseract first (lighter), then Unstructured
        try:
            import pytesseract
            return _load_image_pytesseract(file_path, filename)
        except ImportError:
            return _load_image_unstructured(file_path, filename)
    elif ext in TEXT_EXTENSIONS:
        return _load_text_file(file_path, filename)
    else:
        # Try as text file
        try:
            return _load_text_file(file_path, filename)
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .png, .jpg, .txt, .md, .csv")


# Backward compatibility
def load_pdf(file_path: str) -> list:
    """Load PDF only (legacy)."""
    return _load_pdf(file_path)
