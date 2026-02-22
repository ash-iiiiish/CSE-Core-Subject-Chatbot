from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llm import load_llm
from prompt import get_prompt
from vectorstore import load_vectorstore

def create_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    llm = load_llm()
    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain