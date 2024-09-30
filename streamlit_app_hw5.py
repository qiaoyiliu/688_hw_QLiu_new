import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="~/embeddings")

# OpenAI setup
if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

st.subheader("Step 1: Upload PDFs")
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

# Get or create ChromaDB collection
if "HW5_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="HW5Collection")

# Function to read PDFs
def read_pdf(file):
    file_name = file.name
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
    return file_name, pdf_content

# Function to add PDF content to ChromaDB
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

# Process PDF uploads
if uploaded_files:
    if st.button("Finish Upload and Process PDFs"):
        for file in uploaded_files:
            filename, text = read_pdf(file)
            add_to_collection(st.session_state.HW5_vectorDB, text, filename)
            st.success(f"Document '{filename}' added to the vector DB.")
        st.session_state.pdfs_uploaded = True

# Step 2: Function to retrieve relevant course info based on user's query
def relevant_course_info(location, format, chromadb_collection):
    openai_client = st.session_state.openai_client
    
    # Embed the user query to compare with stored documents
    response = openai_client.embeddings.create(
        input=location,  # Use 'location' as the user's course query
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Query the ChromaDB collection
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top 3 most relevant documents
    )
    
    if not results['documents']:
        st.error("No relevant course information found.")
        return None

    # Display the most relevant course information
    top_course_info = results['documents'][0]
    return top_course_info

# Define tools using function format
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Retrieve relevant course information based on user query",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The course name or question the user has about the course.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],  # Keeping it for structure, can adjust for courses
                        "description": "The format of the answer (useful for structuring the response).",
                    },
                },
                "required": ["location", "format"],
            },
        }
    }
]

GPT_MODEL = "gpt-4o-mini"
client = openai

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# Step 3: Input for user to ask a course-related question
st.subheader("Step 2: Ask about a course")
user_query = st.text_input("Enter your course-related question (e.g., 'Tell me about AI course requirements'): ")

# Button to trigger course information retrieval
if st.button("Get Course Information"):
    if user_query and "HW5_vectorDB" in st.session_state:
        # Messages for LLM interaction
        messages = []
        messages.append({"role": "system", "content": "Answer the user’s course-related questions by retrieving relevant information."})
        messages.append({"role": "user", "content": user_query})

        # Call the LLM with the tool for course info retrieval
        chat_response = chat_completion_request(messages, tools=tools, model=GPT_MODEL)
        
        # Extract the tool response, and pass it as natural language response
        if chat_response.choices[0].message.tool_calls:
            tool_call = chat_response.choices[0].message.tool_calls[0]
            
            # Process the relevant course info using the function call
            course_info = relevant_course_info(location=tool_call.arguments['location'], format=tool_call.arguments['format'], chromadb_collection=st.session_state.HW5_vectorDB)
            
            # Return the course info in natural language
            if course_info:
                st.write(f"The most relevant course information: {course_info}")
            else:
                st.error("Could not retrieve relevant course information.")
    else:
        st.error("Please upload course PDFs first or enter a valid query.")
