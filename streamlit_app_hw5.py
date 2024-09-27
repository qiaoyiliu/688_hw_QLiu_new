import openai
import chromadb
import streamlit as st
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("Joy's HW5 Using functions/tools for course-related chatbot")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="~/embeddings")

if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    #openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key

# Function to read and extract text from PDF
def read_pdf(file):
    file_name = file.name  
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
    return file_name, pdf_content

# Upload PDF files
uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)

# Initialize the vector collection if not done yet
if "HW5_vectorDB" not in st.session_state:
    st.session_state.HW5_vectorDB = chroma_client.get_or_create_collection(name="Lab4Collection")

# Function to add PDF content to the vector database
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

# Add uploaded PDFs to the collection
if uploaded_files and "HW5_vectorDB" in st.session_state:
    for file in uploaded_files:
        filename, text = read_pdf(file)
        add_to_collection(st.session_state.HW5_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")

# Define the initial messages and system prompt
system_message = '''Answer course-related questions using the knowledge gained from the context.'''
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_message},
                                    {"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":    
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

# Assume a message from the model includes a tool call
openai_client = st.session_state.openai_client
response_message = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=st.session_state.messages
)

# Step 2: Check if the model response includes a tool call
tool_calls = response_message.choices[0].tool_calls
if tool_calls:
    # Extract the function name and input needed for the query (raw query text instead of JSON)
    tool_call_id = tool_calls[0].id
    tool_function_name = tool_calls[0].function.name
    tool_query_string = tool_calls[0].function.arguments['query']  # Just use the raw query string

    # Step 3: Execute the appropriate function based on the tool call
    if tool_function_name == 'ask_chromadb':
        query_embedding = openai.embeddings.create(
            input=tool_query_string,
            model="text-embedding-3-small"
        )['data'][0]['embedding']

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
        
        context = "\n\n".join(relevant_documents)

        # Append results as a response to the tool call
        st.session_state.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_function_name,
            "content": context
        })

        # Step 4: Invoke the chat completions API with the function response appended to the messages list
        openai_client = st.session_state.openai_client
        final_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages,
        )

        # Output the final response
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response['choices'][0]['message']['content']
        })

        with st.chat_message("assistant"):
            st.write(final_response['choices'][0]['message']['content'])

    else:
        print(f"Error: function {tool_function_name} does not exist")
else:
    # No tool call; simply return the response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_message.choices[0].message['content']
    })

    with st.chat_message("assistant"):
        st.write(response_message.choices[0].message['content'])
