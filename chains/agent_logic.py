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
    "You are a helpful assistant for iCog Labs. "
    "You have been provided with documents via a RAG (Retrieval-Augmented Generation) system. "
    "Always check the CONTEXT provided below to answer the user. "
    "If the answer is in the context, use it. If not, tell the user you don't see that in the uploaded files.\n\n"
    "CONTEXT:\n{context}"
)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        # MULTIMODAL TIP: You can check docs for image metadata here
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Final Chain (Supports Streaming)
    rag_chain = (
        RunnablePassthrough.assign(
            rewritten_input = question_rewriter 
        )
        | RunnablePassthrough.assign(
            context = lambda x: format_docs(advanced_retriever.invoke(x["rewritten_input"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

