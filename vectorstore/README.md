# 🗄️ Vector Store (Knowledge Base)
This module manages the long-term memory of our RAG system.

### Key Features:
- **ChromaDB Integration:** A high-performance vector database used for similarity search.
- **Persistence:** Configured to save the index to the `chroma_db/` directory so data isn't lost when the server restarts.
- **Optimized Embeddings:** Uses `text-embedding-3-small` for a perfect balance of speed and semantic depth.

### Why this matters:
A RAG system is only as good as its search engine. By using a persistent vector store, we ensure the system is scalable and ready for production environments.