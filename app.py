import streamlit as st
from pymilvus import MilvusClient
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("📚 PDF RAG Chatbot")
@st.cache_resource
def get_milvus_client():
    return MilvusClient(
        uri=st.secrets["MILVUS_URI"],
        token=st.secrets["MILVUS_TOKEN"]
    )
# Connect to the instance
client = get_milvus_client()

client.drop_collection(collection_name="rag_collection")
print("drop success")
