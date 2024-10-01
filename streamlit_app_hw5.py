import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

st.title("Joy's HW5")

# Initialize ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="~/embeddings")

def read_pdf(file):
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content

# Upload PDFs
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Create or retrieve the ChromaDB collection
if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

# Function to add documents to ChromaDB
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

# Add uploaded files to the collection
if uploaded_files is not None and "HW5_vectorDB" in st.session_state:
    for i in uploaded_files:
        filename, text = read_pdf(i)
        add_to_collection(st.session_state.HW5_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")

# Define a tool for generating the embedding and querying ChromaDB
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_chromadb",
            "description": "Retrieve relevant documents from ChromaDB based on a user's query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or query.",
                    }
                },
                "required": ["query"]
            },
        }
    }
]

# Define the function that will be invoked by the tool
def query_chromadb(query):
    openai_client = st.session_state.openai_client
    query_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    # Search for the top 3 relevant documents in the ChromaDB collection
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

# Chatbot orchestration, calling the query tool to get relevant documents
GPT_MODEL = "gpt-4o-mini"
client = openai

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Automatically choose the appropriate tool
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# Initialize system message for chat
system_message = '''Answer course-related questions using the knowledge gained from the context.'''

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":    
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

# User input prompt
if prompt := st.chat_input("What is up?"):
    # Call the tool to handle the embedding generation and document search
    chat_response = chat_completion_request(
        messages=st.session_state.messages,
        tools=tools,
        model=GPT_MODEL
    )

    # Check if the tool 'query_chromadb' was successfully called
    if 'tool_calls' in chat_response.choices[0].message and 'query_chromadb' in chat_response.choices[0].message.tool_calls:
        assistant_message = chat_response.choices[0].message.tool_calls['query_chromadb']
        relevant_documents = assistant_message.get('output', [])

        # Append the user prompt and assistant's answer to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": f"Relevant documents:\n{relevant_documents}"})

        # Display the assistant's response
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(f"Relevant documents:\n{relevant_documents}")
    else:
        # If the tool wasn't called, fall back to generating a general LLM response
        fallback_messages = st.session_state.messages + [
            {"role": "user", "content": prompt},
            {"role": "system", "content": "Generate a general answer without specific document retrieval."}
        ]
        
        general_response = chat_completion_request(
            messages=fallback_messages,
            tools=None,  # No tool required for general LLM generation
            model=GPT_MODEL
        )
        
        general_answer = general_response.choices[0].message['content']
        
        # Append the user prompt and general answer to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": general_answer})

        # Display the general LLM response
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(general_answer)
