import re
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


def _parse_queries(llm_response: str, original: str, max_queries: int = 4) -> list:
    
    queries = [original.strip()]
    lines = llm_response.strip().split("\n")
    for line in lines:
       
        cleaned = re.sub(r"^[\d\)\.\-\*\:\s]+", "", line).strip()
        if cleaned and len(cleaned) > 3 and cleaned.lower() not in (q.lower() for q in queries):
            queries.append(cleaned)
        if len(queries) >= max_queries:
            break
    return queries[:max_queries]


def get_rrf_retriever(llm, vectorstore):
    # Higher k for better recall; RRF will dedupe and rerank
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    
    def multi_query_and_fuse(input_data):
        question = input_data if isinstance(input_data, str) else input_data.get("input", "")
        if not question:
            return []
        
        # Generate query variations
        try:
            query_response = llm.invoke(
                "Generate 2-3 short, distinct search queries for this question. "
                "One per line, no numbering. Question: " + question
            )
            queries = _parse_queries(query_response.content, question)
        except Exception:
            queries = [question]
        
        all_docs = []
        for q in queries:
            if q.strip():
                docs = base_retriever.invoke(q.strip())
                all_docs.append(docs)
        
        if not all_docs:
            return []
        return reciprocal_rank_fusion(all_docs)

    return RunnableLambda(multi_query_and_fuse)