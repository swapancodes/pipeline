import streamlit as st
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("📚 PDF RAG Chatbot")

from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

# 1. Load your data
loader = DirectoryLoader("data/docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
docs = loader.load()
print("success")


