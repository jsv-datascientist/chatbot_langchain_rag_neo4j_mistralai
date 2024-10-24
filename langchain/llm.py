import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPEN_API_KEY"], #can be replaced with mistral api key
    model=st.secrets["OPENAI_MODEL"], #can be replaced with mistral model
)

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)