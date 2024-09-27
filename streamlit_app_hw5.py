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

chroma_client = chromadb.PersistentClient(path="~/embeddings")

if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

st.subheader("Step 1: Upload PDFs")
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

def read_pdf(file):
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
    return file_name, pdf_content

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

if uploaded_files:
    if st.button("Finish Upload and Process PDFs"):
        for file in uploaded_files:
            filename, text = read_pdf(file)
            add_to_collection(st.session_state.HW5_vectorDB, text, filename)
            st.success(f"Document '{filename}' added to the vector DB.")
        st.session_state.pdfs_uploaded = True  

if "pdfs_uploaded" in st.session_state:
    st.subheader("Step 2: Ask Questions")
    
    user_question = st.text_input("Enter your question:")

    if st.button("Submit Question") and user_question:
        query_response = st.session_state.openai_client.embeddings.create(
            input=user_question,
            model="text-embedding-3-small"
        )
        query_embedding = query_response.data[0].embedding
    
        def ask_chromadb(query_embedding, n_results=1):
            """
            Function to query ChromaDB using the provided query embedding.
            Args:
                query_embedding (list): A list representing the embedding vector for the user's query.
                n_results (int): The number of top results to retrieve from the database. Default is 3.
                
            Returns:
                dict: The query results from ChromaDB.
            """
            try:
                results = st.session_state.HW5_vectorDB.query(
                    query_embeddings=[query_embedding],  
                    n_results=n_results                 
                )
                return results
            
            except Exception as e:
                return f"Query failed with error: {e}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant. You have access to a database of documents."},
            {"role": "user", "content": user_question},
            {"role": "system", "content": "You must use the `ask_chromadb` tool to answer this question based on the documents available."}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ask_chromadb",
                    "description": "Use this function to query the document database and retrieve relevant documents based on the user question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_embedding": {
                                "type": "array",
                                "items": {
                                    "type": "number"  
                                },
                                "description": "Embedding vector of the user query."
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of top results to retrieve from the database.",
                                "default": 1
                            }
                        },
                        "required": ["query_embedding"]
                    }
                }
            }
        ]

        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            tools=tools, 
            tool_choice="required"  
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        tool_calls = response_message.tool_calls

        if tool_calls:
            tool_call_id = tool_calls[0].id
            tool_function_name = tool_calls[0].function.name
            tool_arguments = json.loads(tool_calls[0].function.arguments)

            if tool_function_name == 'ask_chromadb':
                query_embedding = tool_arguments['query_embedding']
                n_results = tool_arguments.get('n_results', 1) 
                
                results = ask_chromadb(query_embedding, n_results)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_function_name,
                    "content": results
                })

                model_response_with_function_call = st.session_state.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )

                st.write(model_response_with_function_call.choices[0].message.content)
        else:
            st.write(response_message.content)


