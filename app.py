
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
#creating my prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer using the provided context."),
    ("human", "{question}")
])

# streamlit freamework 
st.title("LangChain demo with gemma2:2b")
input_text = st.text_input("Enter your question here:")


# ollama model
llm = OllamaLLM(model="gemma2:2b")
response_schemas = [
    ResponseSchema(
        name="answer",
        description="The answer to the user's question"
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
chain = prompt | llm | output_parser

chain = prompt | llm

result = chain.invoke({"question": input_text})
st.write(result)
