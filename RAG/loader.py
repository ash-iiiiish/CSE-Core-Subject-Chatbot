from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdfs(pdf_folder_path):
    documents = []
    
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder_path, file))
            documents.extend(loader.load())
    
    return documents