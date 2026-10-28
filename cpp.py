
import streamlit as st
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="C++ LLM Assistant")

st.title("C++ LLM Assistant")
st.write("Ask questions from your cpp.txt file.")

# Read cpp.txt from the same folder as this Python file
file_path = Path(__file__).parent / "cpp.txt"

if not file_path.exists():
    st.error("cpp.txt not found. Please keep it in the same folder as cpp_app.py.")
    st.stop()

cpp_content = file_path.read_text(encoding="utf-8")

# Connect to Gemma through Ollama
llm = OllamaLLM(model="gemma:2b")

prompt = ChatPromptTemplate.from_template("""
You are a helpful C++ tutor.
Answer the question using the following C++ notes.
If the answer is not in the notes, say so clearly.

C++ NOTES:
{context}

QUESTION:
{question}

Give a simple explanation with examples when useful.
""")

chain = prompt | llm

question = st.text_input("Enter your C++ question")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = chain.invoke({
                    "context": cpp_content,
                    "question": question
                })
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
