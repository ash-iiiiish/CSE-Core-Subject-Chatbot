import streamlit as st
from pipeline import create_rag_chain

st.set_page_config(page_title="PDF Chatbot (Groq)")
st.title("📄 PDF Chatbot using Groq")

@st.cache_resource
def load_chain():
    return create_rag_chain()

rag_chain = load_chain()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

user_question = st.chat_input("Ask something from your PDFs...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])