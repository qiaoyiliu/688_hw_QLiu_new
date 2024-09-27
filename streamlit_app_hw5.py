import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import json

st.title("Joy's HW5 Chatbot")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="~/embeddings")

# Ensure OpenAI client is initialized
if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Step 1: Upload PDFs
st.subheader("Step 1: Upload PDFs")
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

# Initialize the vector DB collection if not already done
if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

# Function to read PDF content
def read_pdf(file):
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
    return file_name, pdf_content

# Function to add documents to the ChromaDB collection
def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
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

# Step 2: Finalize PDF upload
if uploaded_files:
    if st.button("Finish Upload and Process PDFs"):
        for file in uploaded_files:
            filename, text = read_pdf(file)
            add_to_collection(st.session_state.HW5_vectorDB, text, filename)
            st.success(f"Document '{filename}' added to the vector DB.")
        st.session_state.pdfs_uploaded = True  # Mark that PDFs are uploaded and processed

# Function to query ChromaDB
def ask_chromadb(query_embedding, n_results=3):
    """Function to query ChromaDB based on embedding."""
    try:
        results = st.session_state.HW5_vectorDB.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    except Exception as e:
        return f"Query failed with error: {e}"

# Step 3: Ask Questions
if "pdfs_uploaded" in st.session_state:
    st.subheader("Step 2: Ask Questions")
    
    user_question = st.text_input("Enter your question:")

    if st.button("Submit Question") and user_question:
        try:
            # Create embedding for the user's question
            query_response = st.session_state.openai_client.embeddings.create(
                input=user_question,
                model="text-embedding-3-small"
            )
            query_embedding = query_response.data[0].embedding
            
            # Query ChromaDB with the user's question embedding
            results = ask_chromadb(query_embedding)

            if isinstance(results, str):
                st.error(results)  # Show error if query failed
            else:
                # Process and display the results
                top_docs = [f"Document ID: {res['id']} - Content: {res['document']}" for res in results['documents']]
                st.write("Top 3 relevant documents:")
                for doc in top_docs:
                    st.write(doc)
        
        except openai.error.InvalidRequestError as e:
            st.error(f"Failed to process the question: {e}")
else:
    st.warning("Please upload and process PDFs first before asking questions.")
