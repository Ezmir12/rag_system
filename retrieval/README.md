# 🔍 Advanced Retrieval
This module implements a multi-step search strategy to find the most relevant information.

### Key Features:
- **Multi-Query Generation:** The system rewrites the user's question into 3 different versions to capture different linguistic nuances.
- **Reciprocal Rank Fusion (RRF):** An advanced algorithm that merges results from multiple searches and reranks them based on relevance.

### Why this matters:
Users often ask vague questions. Multi-query expansion ensures we find the right data even if the user didn't use the exact keywords found in the PDF. RRF then ensures the absolute best results rise to the top.