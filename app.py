import streamlit as st
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("📚 PDF RAG Chatbot")

from langchain_community.document_loaders import (DirectoryLoader, PyMuPDFLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
COLLECTION_NAME = "demo_collection"

#-----------Milvus Connection-----------
@st.cache_resource
def get_milvus_client():
    return MilvusClient(
        uri=st.secrets["MILVUS_URI"],
        token=st.secrets["MILVUS_TOKEN"]
    )

#----------Load and Chunk PDFs-------------
@st.cache_data
def load_documents():
    loader = DirectoryLoader(
        "data/docs",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return chunks

#----------Generate Embeddings---------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vector_store():
    client = get_milvus_client()
    chunks = load_documents()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embedding_model()
    vectors = embeddings.embed_documents(texts)
    # Create collection only once
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=384
        )
        data = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            data.append(
                {
                    "id": i,
                    "vector": vector,
                    "text": chunk.page_content,
                }
            )
        client.insert(
            collection_name=COLLECTION_NAME,
            data=data
        )

        st.success(
            f"Inserted {len(data)} chunks into Milvus."
        )
        st.success(
            f"seceess full inserted."
        )
        st.success(
            f"Inserted {len(vectors)} vectore."
        )
    else:
        st.info("Collection already exists.")

#---------Main---------
try:
    build_vector_store()
except Exception as e:
    st.error(f"Error: {str(e)}")
