import streamlit as st
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ----------------------------
# Load .env file
# ----------------------------
load_dotenv()

# Optional: debug check
# st.write(os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="PDF Chatbot (Groq)")
st.title("📄 PDF Chatbot using Groq")

# ----------------------------
# Load Vector Store
# ----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "C:/Users/kumar/OneDrive/Desktop/TRY-4/CSE-Core-Subject-Chatbot/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# ----------------------------
# Groq LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant.
Answer ONLY from the provided context.

Context:
{context}

Question:
{question}
""")

# ----------------------------
# Format Documents
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ----------------------------
# RAG Chain
# ----------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------------------
# Chat UI
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_question = st.chat_input("Ask something from your PDFs...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)

    st.session_state.messages.append({"role": "assistant", "content": response})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])