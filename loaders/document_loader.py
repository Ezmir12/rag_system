from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path):
    """
    Loads a PDF and prepares it for advanced chunking.
    Includes error handling for missing files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")

    try:
        # PyPDFLoader reads the PDF and splits it by physical pages initially
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Adding a logic check: ensure we actually got text
        if not documents:
            print("Warning: The PDF appears to be empty or image-only.")
            
        return documents
    
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []