import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import json

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
    results = st.session_state.HW5_vectorDB.query(
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
            tools=tools,
            tool_choice="auto",  
        )
        response_message = response.choices[0].message
        tool_calls = response_message.get("tool_calls", [])

        if tool_calls:
            # Extract tool call details
            tool_calls_id = tool_calls[0].get("id", "")
            tool_function_name = tool_calls[0].get("function", {}).get("name", "")
            tool_arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
            tool_query_string = json.loads(tool_arguments).get('query', "")

            if tool_function_name == "query_chromadb":
                # Assuming `query_chromadb` is a function that handles the ChromaDB query
                results = query_chromadb(tool_query_string)

                # Append tool-based answer to the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Results from ChromaDB query:\n{results}"
                })

                # Display the tool's results in the UI
                with st.chat_message("assistant"):
                    st.write(f"Results from ChromaDB query:\n{results}")
            else:
                # Handle other tool calls here if necessary
                pass
        else:
            # No tool call was made, generate a general answer using the LLM
            fallback_messages = st.session_state.messages + [
                {"role": "user", "content": prompt},
                {"role": "system", "content": "Generate a general answer without specific document retrieval."}
            ]
            
            # Generate a general response without tools
            general_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=fallback_messages,
                tools=None,  # No tool required for general LLM generation
                tool_choice=None  # Ensure tool-free LLM generation
            )

            general_answer = general_response.choices[0].message['content']

            # Append the general answer to the chat history
            st.session_state.messages.append({"role": "assistant", "content": general_answer})

            # Display the general answer
            with st.chat_message("assistant"):
                st.write(general_answer)