# 🧠 Semantic Processing
This module is responsible for breaking down large documents into manageable pieces.

### Key Features:
- **Semantic Chunking:** Unlike standard "Fixed-Size" splitting, this uses **Embeddings** to find natural breakpoints where the meaning of the text changes.
- **Context Preservation:** Ensures that sentences belonging to the same topic stay together.

### Why this matters:
iCog Labs requires an "Advanced" system. Standard splitting often cuts a sentence in half, causing the AI to lose context. Semantic chunking ensures the "thought" is preserved, leading to significantly higher retrieval accuracy.