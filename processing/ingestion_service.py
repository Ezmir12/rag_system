from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def process_documents(documents, embeddings_model):
    """
    Implements Advanced Semantic Chunking as per iCog requirements.
    """
    if not documents:
        return []

    # 'percentile' threshold identifies shifts in topic automatically
    splitter = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="percentile"
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Successfully created {len(chunks)} semantic chunks.")
    return chunks





