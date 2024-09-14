import streamlit as st
import openai
import os
from openai import OpenAI
import pypdf
import pdfplumber
from anthropic import Anthropic
from anthropic.types.message import Message
from mistralai import Mistral
import requests
from bs4 import BeautifulSoup

st.title("Joy's Document question answering for HW2")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an API key."
)

# Read PDF files
def read_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Read URL content
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# Sidebar for selecting model
selected_llm = st.sidebar.selectbox("Which model?",
                                    ("gpt-4o-mini", "gpt-4o",
                                    "claude-haiku", "claude-opus",
                                    "mistral-small", "mistral-medium"))

client = None  # Initialize client as None

# Model and API key selection
if selected_llm in ['gpt-4o-mini', 'gpt-4o']:
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Please provide OpenAI API key")
elif selected_llm in ['claude-haiku', 'claude-opus']:
    api_key = st.text_input("Anthropic API Key", type="password")
    if api_key:
        client = Anthropic(api_key=api_key)
    else:
        st.warning("Please provide Anthropic API key")
elif selected_llm in ['mistral-small', 'mistral-medium']:
    api_key = st.text_input("Mistral API Key", type="password")
    if api_key:
        client = Mistral(api_key=api_key)
    else:
        st.warning("Please provide Mistral API key")

# Check if the client is initialized
if client is None:
    st.error("Client is not initialized. Please check the API key.")
else:
    # File uploader and URL input
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, or .pdf)", type=("txt", "pdf")
    )

    # Ask user to paste URL
    question_url = st.text_area(
        "Or insert a URL:",
        placeholder="Copy URL here",
    )

    # Ask a question
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file and not question_url,
    )

    # Language selection
    languages = ['English', 'Spanish', 'French']
    selected_language = st.selectbox('Select your language:', languages)
    st.write(f"You have selected: {selected_language}")

    # Prepare the document or URL content and messages
    if uploaded_file and question:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
        messages = [
            {
                "role": "user",
                "content": f"Respond in {selected_language}. Here's a document: {document} \n\n---\n\n {question}",
            }
        ]
    elif question_url:
        url_content = read_url_content(question_url)
        messages = [
            {
                "role": "user",
                "content": f"Respond in {selected_language}. Here's a URL: {url_content} \n\n---\n\n {question}",
            }
        ]

    # Handling LLM selection
    if selected_llm == "gpt-4o-mini":
        stream = client.chat.completions.create(
            model=selected_llm,
            max_tokens=250,
            messages=messages,
            stream=True,
            temperature=0.5,
        )
        st.write_stream(stream)

    elif selected_llm == 'claude-haiku':
        message = client.messages.create(
            model=selected_llm,
            max_tokens=256,
            messages=messages,
            temperature=0.5,
        )
        data = message.content[0].text
        st.write(data)

    elif selected_llm == 'mistral-small':
        response = client.chat.complete(
            model=selected_llm,
            max_tokens=250,
            messages=messages,
            temperature=0.5,
        )
        data = response.choices[0].message.content
        st.write(data)
