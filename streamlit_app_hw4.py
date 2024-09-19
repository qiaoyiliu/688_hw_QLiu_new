import streamlit as st
from openai import OpenAI
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

st.title("Joy's Lab4 question answering chatbot")

topic = st.sidebar.selectbox("Topic", ("Generative AI", "Text Mining", "Data Science Overview"))

#chroma_client = chromadb.PersistentClient(path="~/embeddings")
chroma_client = chromadb.Client()

def read_pdf(file):
    file_name = file.name  
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content

uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf"))

if "client" not in st.session_state:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.session_state.openai_client = OpenAI(api_key=openai_api_key)

    if "Lab4_vectorDB" not in st.session_state:
        st.session_state.Lab4_vectorDB = chroma_client.get_or_create_collection(name="Lab4Collection")
        
        def add_to_collection(collection, text, filename):
            openai_client = st.session_state.client
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            collection.add(
                documents=[text],
                ids=[filename],
                embeddings=[embedding]
            )

        if uploaded_file is not None:
            filename, text = read_pdf(uploaded_file)
            
            add_to_collection(st.session_state.Lab4_vectorDB, text, filename)
            st.success(f"Document {filename} added to the vector DB.")

        openai_client = st.session_state.openai_client
        response = chroma_client.embeddings.create(
        input=topic,
        model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = st.session_state.Lab4_vectorDB.query(
            query_embedding=[query_embedding],
            n_results=3
        )

        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            doc_id = results['ids'][0][i]
            st.write(f"The following file/syllabus might be helpful: {doc_id}")


