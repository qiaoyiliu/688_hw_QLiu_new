import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

st.title("Joy's HW4 Question Answering Chatbot")

# Initialize ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="~/embeddings")

# Function to read and extract text from PDF
def read_pdf(file):
    file_name = file.name  
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content

# Upload PDF documents
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

# Initialize OpenAI client
if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Initialize ChromaDB collection for HW4
if "HW4_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW4_vectorDB = chroma_client.get_or_create_collection(name="HW4Collection")

# Function to add PDF content to ChromaDB collection
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

# Add uploaded files to ChromaDB collection
if uploaded_files is not None and "HW4_vectorDB" in st.session_state:
    for i in uploaded_files:
        filename, text = read_pdf(i)
        add_to_collection(st.session_state.HW4_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")

openai_client = st.session_state.openai_client
     
# System message for the chatbot
system_message = '''Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.'''

# Store chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_message}
    ]

# Function to search documents in ChromaDB using query embeddings
def search_documents_in_chromadb(query_embedding):
    results = st.session_state.HW4_vectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    relevant_documents = []
    if results and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            relevant_text = results['documents'][0][i]
            relevant_documents.append(relevant_text)
    else:
        relevant_documents = ["No relevant documents found."]
    
    return relevant_documents

# Define the tool that the LLM can use
tools = [
    {
        "name": "search_documents_in_chromadb",
        "description": "Search for relevant documents in the ChromaDB using embeddings",
        "parameters": {
            "type": "object",
            "properties": {
                "query_embedding": {
                    "type": "array",
                    "description": "The embedding of the user's query to search relevant documents"
                }
            },
            "required": ["query_embedding"]
        }
    }
]

# Function to call LLM with tools (ChromaDB document retrieval) and clarification behavior
def call_llm_with_tools(messages, query_embedding=None):
    # If query_embedding exists, use the function call, else just ask for clarification
    function_call_data = {"name": "search_documents_in_chromadb"} if query_embedding else "auto"

    response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Use the appropriate model
        messages=messages,
        functions=tools,
        function_call=function_call_data  # Auto call function only if query_embedding exists
    )

    return response['choices'][0].get('message'), response['choices'][0].get('function_call')

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] != "system":    
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

# Get user input and process the query
if prompt := st.chat_input("What is up?"):
    # Add user input to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create the query embedding for the input
    query_response = openai_client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    # Call the LLM with tools for document retrieval or clarification
    assistant_message, function_call = call_llm_with_tools(st.session_state.messages, query_embedding)

    # Add assistant message to the conversation
    if assistant_message:
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(assistant_message["content"])

    # Handle if function_call is made
    if function_call and function_call['name'] == "search_documents_in_chromadb":
        relevant_documents = search_documents_in_chromadb(query_embedding)
        context = "\n\n".join(relevant_documents)
        st.write(f"Relevant documents found:\n\n{context}")

        # Provide the new context and prompt for the next stage
        st.session_state.messages.append({"role": "system", "content": f"Relevant documents: {context}"})
