import streamlit as st
import requests
from bs4 import BeautifulSoup

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'questions_buffer' not in st.session_state:
    st.session_state.questions_buffer = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'url_content' not in st.session_state:
    st.session_state.url_content = None

# Sidebar to choose the model
model_choice = st.sidebar.selectbox(
    "Choose model:",
    ("gpt-4o-mini", "gpt-4o")
)

# Function to update the question buffer
def update_question_buffer(question):
    st.session_state.questions_buffer.append(question)
    # Keep only the last 5 questions
    st.session_state.questions_buffer = st.session_state.questions_buffer[-5:]

# Function to read content from a URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        return f"Error reading {url}: {e}"

# API Key input
if st.session_state.api_key is None:
    st.session_state.api_key = st.text_input("Insert your API key:", type="password")
    if st.button("Save API Key"):
        st.success("API key saved.")
else:
    st.write("API Key: **Saved**")

# Upload URL
url_input = st.text_input("Enter a URL to summarize:")
if st.button("Upload URL"):
    if url_input:
        st.session_state.url_content = read_url_content(url_input)
        if st.session_state.url_content:
            st.write("URL content uploaded successfully.")

# Chat interface
if st.session_state.url_content:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Capture user input from chat
    if prompt := st.chat_input("Ask anything or request a summary of the URL:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the question to the buffer
        update_question_buffer(prompt)

        # Summarize the URL if requested
        if "summarize" in prompt.lower():
            summary = st.session_state.url_content[:500] + "..."  # Simple placeholder summary
            with st.chat_message("assistant"):
                st.markdown(f"Summary of the URL: {summary}")
            st.session_state.messages.append({"role": "assistant", "content": summary})
        else:
            # Placeholder for model response (replace with actual API call)
            response = f"Simulated response to: {prompt}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display the last 5 questions
    if st.session_state.questions_buffer:
        st.write("Last 5 questions:")
        for i, question in enumerate(st.session_state.questions_buffer, 1):
            st.write(f"{i}. {question}")

