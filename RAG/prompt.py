from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_template("""
You are a helpful PDF assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say:
"I could not find this information in the provided PDFs."

Context:
{context}

Question:
{question}
""")