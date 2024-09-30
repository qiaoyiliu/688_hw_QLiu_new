import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# ChromaDB client setup
chroma_client = chromadb.PersistentClient(path="~/embeddings")

# OpenAI API key setup
if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Upload PDFs and Create ChromaDB Collection
st.subheader("Step 1: Upload PDFs")
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

# Function to read the PDF file
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

# Upload and process PDF files
if uploaded_files:
    if st.button("Finish Upload and Process PDFs"):
        for file in uploaded_files:
            filename, text = read_pdf(file)
            add_to_collection(st.session_state.HW5_vectorDB, text, filename)
            st.success(f"Document '{filename}' added to the vector DB.")
        st.session_state.pdfs_uploaded = True

# Tool function for retrieving relevant course information from ChromaDB
def relevant_course_info(location):
    # Query ChromaDB to retrieve relevant course details
    collection = st.session_state.HW5_vectorDB
    results = collection.query(
        query_texts=[location],
        n_results=3  # Retrieve top 3 relevant documents
    )
    
    if results['documents']:
        return results['documents'][0]  # Return the most relevant course information
    else:
        return "No relevant course found."

# Defining tools for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Retrieve relevant course information from the ChromaDB collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The course name or topic, e.g., 'AI requirements'."
                    }
                },
                "required": ["location"]
            },
        }
    }
]

# Chat completion request function
GPT_MODEL = "gpt-4o-mini"  # Update to use 'gpt-4o-mini'

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=tools,
            function_call="auto"
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# Initialize conversation messages
messages = []
messages.append({"role": "system", "content": "You can retrieve course information from uploaded PDFs."})

# Example question from the user
messages.append({"role": "user", "content": "Tell me more about AI course requirements"})

# Send chat request to the model and retrieve response
chat_response = chat_completion_request(
    messages=messages,
    tools=tools,
    model=GPT_MODEL
)

# Check if the LLM makes a tool call
if "function_call" in chat_response.choices[0].message:
    tool_call = chat_response.choices[0].message.function_call
    # Execute the tool and retrieve course info
    course_info = relevant_course_info(location=tool_call['arguments']['location'])
    
    # Pass course info back to LLM for natural language response
    messages.append({"role": "assistant", "content": f"The details for {tool_call['arguments']['location']} are: {course_info}"})
    
    # Generate final natural language response
    chat_response = chat_completion_request(
        messages=messages,
        tools=tools,
        model=GPT_MODEL
    )
    assistant_message = chat_response.choices[0].message.content
    st.write(assistant_message)
else:
    st.write("No tool call was made.")
