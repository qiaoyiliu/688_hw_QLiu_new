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
st.write(
    "Example question I used: "
    "Q1: which of the courses teach AI?"
    "Q2: Tell me more about the first one."
)

chroma_client = chromadb.PersistentClient(path="~/embeddings")

def read_pdf(file):
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content


uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key


if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")


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


if uploaded_files is not None and "HW5_vectorDB" in st.session_state:
    for i in uploaded_files:
        filename, text = read_pdf(i)
        add_to_collection(st.session_state.HW5_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")


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


def query_chromadb(query):
    openai_client = st.session_state.openai_client
    query_response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

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


system_message = '''Answer course-related questions using the knowledge gained from the context.'''

if "messages" not in st.session_state:
    st.session_state["messages"] = \
        [{"role": "assistant", "content": "How can I help you?"}]


for msg in st.session_state.messages:
    if msg["role"] != "system":    
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    
    openai_client = st.session_state.openai_client
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto",  
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        tool_function_name = tool_calls[0].function.name
        tool_arguments = json.loads(tool_calls[0].function.arguments)

        if tool_function_name == 'query_chromadb':
            results = query_chromadb(tool_arguments['query'])
            
            text = "\n\n".join(results)

            
            system_message = f"""
            The user asked the following question: {tool_arguments['query']}
            
            Below is the relevant information extracted from the course materials:

            {text}
            
            Use this information to generate a concise response.

            Clarify if you are using information from relevant course materials.
            """

            st.session_state.messages.append({"role": "system", "content": system_message})

            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
                stream=True
            )

            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        with st.chat_message("assistant"):
            st.write(response_message.content)
        st.session_state.messages.append({"role": "assistant", "content": response_message.content})