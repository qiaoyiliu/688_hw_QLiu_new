import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

st.title("Joy's HW4 Question Answering Chatbot")

# Sidebar for selecting topics
topic = st.sidebar.selectbox("Topic", ("Generative AI", "Text Mining", "Data Science Overview"))

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="~/embeddings")
#chroma_client = chromadb.Client()

# PDF Reading Function
def read_pdf(file):
    file_name = file.name  
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content

# File Uploader
uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf"))

# Initialize OpenAI Client if not already in session state
if "openai_client" not in st.session_state:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Initialize ChromaDB Collection if not already in session state
if "Lab4_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.Lab4_vectorDB = chroma_client.get_or_create_collection(name="Lab4Collection")

# Function to Add PDF Content to ChromaDB Collection
def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response['data'][0]['embedding']
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )

# Process Uploaded File and Store Content in Vector DB
if uploaded_file is not None and "Lab4_vectorDB" in st.session_state:
    filename, text = read_pdf(uploaded_file)
    add_to_collection(st.session_state.Lab4_vectorDB, text, filename)
    st.success(f"Document '{filename}' added to the vector DB.")

# Handle Question Based on Sidebar Option
if "openai_client" in st.session_state:
    question = topic
    
    if question:
        query_response = st.session_state.openai_client.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        )
        query_embedding = query_response['data'][0]['embedding']

        results = st.session_state.Lab4_vectorDB.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        st.write("Top 3 relevant documents:")
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            doc_id = results['ids'][0][i]
            st.write(f"The following file/syllabus might be helpful: {doc_id}")
