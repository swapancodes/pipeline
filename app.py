import streamlit as st
from rag_engine import ask_question
st.title("📚 PDF RAG Chatbot")

question = st.text_input("Ask a question")

if st.button("Submit") and question:
    answer = ask_question(question)
    st.write(answer)