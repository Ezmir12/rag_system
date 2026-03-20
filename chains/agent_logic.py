from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retrieval.retrieval_service import get_rrf_retriever

def build_advanced_chain(llm, vectorstore):
    advanced_retriever = get_rrf_retriever(llm, vectorstore)

    # 1. Question Rewriter (Contextualizer)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question that can be understood without history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_rewriter = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Answer Generator
   
    qa_system_prompt = (
        "You are a helpful assistant for iCog Labs. Use the CONTEXT below to answer the user's question. "
        "When the context contains relevant information, provide a clear, accurate answer based on it. "
        "If the context is empty or clearly unrelated to the question, say: 'I couldn't find relevant information about that in the uploaded documents.' "
        "Do not claim information is missing if the context could reasonably support an answer. "
        "Be concise but thorough.\n\n"
        "CONTEXT:\n{context}\n\n"
        "Now answer the user's question based on the context above."
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

    def retrieve_with_fallback(x):
        """Retrieve docs; if empty, fallback to direct search with original question."""
        rewritten = x.get("rewritten_input") or x.get("input", "")
        docs = advanced_retriever.invoke(rewritten)
        if not docs:
            # Fallback: try original question with base retriever
            base = vectorstore.as_retriever(search_kwargs={"k": 8})
            docs = base.invoke(x.get("input", rewritten))
        return format_docs(docs)

    # 3. Final Chain (Supports Streaming)
    rag_chain = (
        RunnablePassthrough.assign(rewritten_input=question_rewriter)
        | RunnablePassthrough.assign(context=retrieve_with_fallback)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

