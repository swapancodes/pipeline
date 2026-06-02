import streamlit as st
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("📚 PDF RAG Chatbot")

from pymilvus import MilvusClient
# Cache the connection pool for performance optimization
@st.cache_resource
def get_milvus_client():
    return MilvusClient(
        uri=st.secrets["MILVUS_URI"],
        token=st.secrets["MILVUS_TOKEN"]
    )
# Connect to the instance
client = get_milvus_client()

st.title("Milvus Vector Database Connection")

#---------Create a Collection----------
client.drop_collection(collection_name="demo1_collection")

# Verify connection by listing available collections
try:
    collections = client.list_collections()
    st.success("Successfully connected to Milvus!")
    st.write("Available Collections:", collections)
except Exception as e:
    st.error(f"Failed to connect to Milvus: {e}")
