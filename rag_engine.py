from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymilvus import MilvusClient
import streamlit as st

api_key = st.secrets("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

#-----------Milvus Connection-----------
@st.cache_resource
def get_milvus_client():
    return MilvusClient(
        uri=st.secrets["MILVUS_URI"],
        token=st.secrets["MILVUS_TOKEN"]
    )
client=get_milvus_client()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}
"""
    ),
    ("human", "{question}")
])

# Create chain
chain = prompt | llm


def ask_question(question: str):
    # Generate embedding
    query_vector = embeddings.embed_query(question)

    # Search Milvus
    results = client.search(
        collection_name="demo_collection",
        data=[query_vector],
        limit=5,
        search_params={"metric_type": "COSINE"},
        output_fields=["text"]
    )

    # Extract text
    retrieved_docs = []

    for hit in results[0]:
        text = hit["entity"]["text"]
        retrieved_docs.append(text)

    context = "\n\n".join(retrieved_docs)

    # Invoke LLM
    response = chain.invoke({
        "context": context,
        "question": question
    })
    return response.content