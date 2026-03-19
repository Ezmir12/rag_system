
import json
from langchain_core.load import dumps, loads
from langchain_core.runnables import RunnableLambda

def reciprocal_rank_fusion(results: list, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return [doc for doc, score in reranked_results]

def get_rrf_retriever(llm, vectorstore):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def multi_query_and_fuse(input_data):
        # Handle string or dict input
        question = input_data if isinstance(input_data, str) else input_data.get("input")
        
        # Generate 3 variations
        query_response = llm.invoke(f"Generate 3 short, distinct search queries for: {question}")
        queries = query_response.content.split("\n")
        
        all_docs = []
        for q in queries:
            if q.strip():
                all_docs.append(base_retriever.invoke(q))
        
        return reciprocal_rank_fusion(all_docs)

    return RunnableLambda(multi_query_and_fuse)


"""import json
from langchain_core.load import dumps, loads
from langchain_core.runnables import RunnableLambda

def reciprocal_rank_fusion(results: list, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return [doc for doc, score in reranked_results]

def get_rrf_retriever(llm, vectorstore):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def multi_query_and_fuse(input_data):
        # Handle string or dict input
        question = input_data if isinstance(input_data, str) else input_data.get("input")
        
        # Generate 3 variations
        query_response = llm.invoke(f"Generate 3 short, distinct search queries for: {question}")
        queries = query_response.content.split("\n")
        
        all_docs = []
        for q in queries:
            if q.strip():
                all_docs.append(base_retriever.invoke(q))
        
        return reciprocal_rank_fusion(all_docs)

    return RunnableLambda(multi_query_and_fuse)
    """