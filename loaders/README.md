# 📥 Document Loaders
This module handles the initial ingestion of raw data into the RAG pipeline.

### Key Features:
- **PDF Extraction:** Utilizes `PyPDFLoader` to extract text from multi-page documents.
- **Metadata Injection:** Automatically tags each extracted page with its original filename (`source`).
- **Validation:** Includes error handling for corrupted or image-only (non-OCR) PDFs.

### Why this matters:
Proper loading is the foundation of a RAG system. By injecting the filename into the metadata at this stage, we allow the final AI agent to cite its sources and tell the user exactly which file it is reading from.