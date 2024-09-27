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
    #openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key


def read_pdf(file):
    """Extract text content from the uploaded PDF."""
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
    return file_name, pdf_content


uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

if "HW4_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW4_vectorDB = chroma_client.get_or_create_collection(name="Lab4Collection")

def add_to_collection(collection, text, filename):
    """Add document content to ChromaDB collection."""
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


if uploaded_files is not None and "HW4_vectorDB" in st.session_state:
    for file in uploaded_files:
        filename, text = read_pdf(file)
        add_to_collection(st.session_state.HW4_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")


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
                            "type": "number"  # This specifies that the array items are numbers (the embedding values)
                        },
                        "description": "Embedding vector of the user query."
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of top results to retrieve from the database.",
                        "default": 3
                    }
                },
                "required": ["query_embedding"]
            }
        }
    }
]


def ask_chromadb(query_embedding, n_results=3):
    """Function to query ChromaDB based on embedding."""
    try:
        results = st.session_state.HW4_vectorDB.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return json.dumps(results)
    except Exception as e:
        return f"Query failed with error: {e}"



messages = [{
    "role": "user", 
    "content": "What are the top 3 relevant documents for my question?"
}]

openai_client = st.session_state.openai_client
query_response = openai_client.embeddings.create(
    input=messages[-1]['content'],
    model="text-embedding-3-small"
)
query_embedding = query_response.data[0].embedding

response = openai_client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=messages, 
    tools=tools, 
    tool_choice="auto"  
)


response_message = response.choices[0].message
messages.append(response_message)  

if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
    tool_calls = response_message.tool_calls
    tool_call_id = tool_calls[0].id
    tool_function_name = tool_calls[0].function.name
    tool_arguments = json.loads(tool_calls[0].function.arguments)

    
    if tool_function_name == "ask_chromadb":
        query_embedding = tool_arguments['query_embedding']
        n_results = tool_arguments.get('n_results', 3)
        results = ask_chromadb(query_embedding, n_results)

        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_function_name,
            "content": results
        })

        
        model_response_with_tool_call = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        
        print(model_response_with_tool_call.choices[0].message.content)

    else:
        print(f"Error: Function {tool_function_name} does not exist")
else:
    print(response_message.content)
