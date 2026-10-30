import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

#page configuration
st.title("C++ RAG Assistant using LLM")

#load and process data
@st.cache_resource
def load_vectorstore():
    # 1. Load document
    loader = TextLoader("CPP.txt", encoding="utf-8")
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create FAISS vector store
    db = FAISS.from_documents(final_documents, embeddings)

    # 5. Return the vector store
    return db

db = load_vectorstore()

# load LLM Ollama

llm = OllamaLLM(model="gemma2:2b")

#chat interface
user_qustion = st.text_input("Ask a question about C++:")

if user_qustion:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(user_qustion, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Answer using ONLY the document context below.

        context:
        {context}
        question:
        {user_qustion}
        Answer:
        """

        response = llm.invoke(prompt)
        st.subheader("Answer:")
        st.write(response)