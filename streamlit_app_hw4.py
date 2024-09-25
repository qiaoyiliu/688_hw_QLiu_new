import streamlit as st
import openai
import os
import pdfplumber
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

st.title("Joy's HW4 Question Answering Chatbot")

chroma_client = chromadb.PersistentClient(path="~/embeddings")
#chroma_client = chromadb.Client()

def read_pdf(file):
    file_name = file.name  
    pdf_content = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text()
        return file_name, pdf_content

uploaded_files = st.file_uploader("Upload a document (.pdf)", type=("pdf"), accept_multiple_files=True)


if "openai_client" not in st.session_state:
    openai_api_key = st.secrets['openai_key']
    #openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state.openai_client = openai
        openai.api_key = openai_api_key


if "Lab4_vectorDB" not in st.session_state and "openai_client" in st.session_state:
    st.session_state.Lab4_vectorDB = chroma_client.get_or_create_collection(name="Lab4Collection")


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


if uploaded_files is not None and "Lab4_vectorDB" in st.session_state:
    for i in uploaded_files:
        filename, text = read_pdf(i)
        add_to_collection(st.session_state.Lab4_vectorDB, text, filename)
        st.success(f"Document '{filename}' added to the vector DB.")

openai_client = st.session_state.openai_client
     

system_message = '''
You are an expert assistant for answering course-related questions.
be clear if it you are using the knowledge gained from the context
'''

if "messages" not in st.session_state:
    st.session_state["messages"] = \
    [{"role": "system", "content": system_message},
     {"role": "assistant", "content": "How can I help you?"}]
    
for msg in st.session_state.messages:
    if msg["role"] != "system":    
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):

    query_response = openai_client.embeddings.create(
    input=prompt,
    model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding

    # Search for the top 3 relevant documents in the ChromaDB
    results = st.session_state.Lab4_vectorDB.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
    
    relevant_documents = ["a", "a"]
    if results and len(results['documents'][0]) > 0:
        relevant_documents = []
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            relevant_text = results['documents'][0][i]  # Text of the document
            relevant_documents.append(relevant_text)
    else:
        relevant_documents = ["No relevant documents found."]
    

    context = "\n\n".join(relevant_documents)

    prompts = f"""
    The user asked: {prompt}
    
    Here is the relevant information from the course documents:

    {context}
    
    Based on this information, please provide a detailed answer.
    """
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "system", "content": prompts})

    with st.chat_message("user"):
        st.markdown(prompt)


    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})