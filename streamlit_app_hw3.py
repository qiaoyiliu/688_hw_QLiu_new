import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
from bs4 import BeautifulSoup
import requests

st.title("Joy's HW3 Multi-LLM Chatbot with URL Summarization")

def summarize_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.text for para in paragraphs[:30]])  
        return content[:2000] 
    except Exception as e:
        return "There was an error fetching the URL content."

def summarize_with_llm(content, selected_llm):
    prompt = f"Please summarize the following content:\n\n{content}"

    if selected_llm == "gpt-4o-mini":
        data = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return data.choices[0].message.content

    elif selected_llm == "gpt-4o":
        data = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return data.choices[0].message.content

    elif selected_llm == 'claude-3-haiku':
        message = client.messages.create(
            model='claude-3-haiku-20240307',
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return message.content[0].text

    elif selected_llm == 'claude-3-opus':
        message = client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return message.content[0].text

    elif selected_llm == 'mistral-small':
        response = client.chat.complete(
            model='mistral-small-latest',
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content

    elif selected_llm == 'mistral-medium':
        response = client.chat.complete(
            model='mistral-medium-latest',
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content

selected_llm = st.sidebar.selectbox(
    "Choose an LLM model:",
    ("gpt-4o-mini", "gpt-4o", "claude-3-haiku", "claude-3-opus", "mistral-small", "mistral-medium")
)

url_option = st.sidebar.radio(
    "Choose the number of URLs to summarize:",
    ("1 URL", "2 URLs")
)

if selected_llm in ['gpt-4o-mini', 'gpt-4o']:
    api_key = st.secrets['OPENAI_API_KEY']
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        st.warning("Please provide OpenAI API key")

elif selected_llm in ['claude-3-haiku', 'claude-3-opus']:
    api_key = st.secrets['ANTHROPIC_API_KEY']
    if api_key:
        client = Anthropic(api_key=api_key)
    else:
        st.warning("Please provide Anthropic API key")

elif selected_llm in ['mistral-small', 'mistral-medium']:
    api_key = st.secrets['MISTRAL_API_KEY']
    if api_key:
        client = Mistral(api_key=api_key)
    else:
        st.warning("Please provide Mistral API key")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'url_summary_1' not in st.session_state:
    st.session_state['url_summary_1'] = None
if 'url_summary_2' not in st.session_state:
    st.session_state['url_summary_2'] = None
if 'summary_added' not in st.session_state:
    st.session_state['summary_added'] = False

#1st url
question_url_1 = st.text_area("Insert the first URL:", placeholder="Copy the first URL here")
if question_url_1 and not st.session_state['summary_added']:
    url_content = summarize_url(question_url_1)  
    st.session_state['url_summary_1'] = summarize_with_llm(url_content, selected_llm) 
    if st.session_state['url_summary_1']:
        st.session_state['messages'].insert(0, {
            "role": "system",
            "content": f"Summary of the first URL: {st.session_state['url_summary_1']}"
        })
        st.session_state['summary_added'] = True

#2nd url
if url_option == "2 URLs":
    question_url_2 = st.text_area("Insert the second URL:", placeholder="Copy the second URL here")
    if question_url_2 and not st.session_state.get('summary_added_2', False):
        url_content = summarize_url(question_url_2)  
        st.session_state['url_summary_2'] = summarize_with_llm(url_content, selected_llm)  
        if st.session_state['url_summary_2']:
            st.session_state['messages'].insert(1, {
                "role": "system",
                "content": f"Summary of the second URL: {st.session_state['url_summary_2']}"
            })
            st.session_state['summary_added_2'] = True

memory_option = st.sidebar.radio(
    "Choose how to store memory:",
    ("Last 5 questions", "Summary of entire conversation", "Last 5,000 tokens")
)

for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    # Append user input to session state
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state['messages']]

    if selected_llm == "gpt-4o-mini":
        data = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=250,
            messages=messages,
            stream=False,
            temperature=0.5,
        )
        response_content = data.choices[0].message.content

    elif selected_llm == "gpt-4o":
        data = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=250,
            messages=messages,
            stream=False,
            temperature=0.5,
        )
        response_content = data.choices[0].message.content

    elif selected_llm == 'claude-3-haiku':
        message = client.messages.create(
            model='claude-3-haiku-20240307',
            max_tokens=256,
            messages=messages,
            temperature=0.5
        )
        response_content = message.content[0].text

    elif selected_llm == 'claude-3-opus':
        message = client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=256,
            messages=messages,
            temperature=0.5
        )
        response_content = message.content[0].text

    elif selected_llm == 'mistral-small':
        response = client.chat.complete(
            model='mistral-small-latest',
            max_tokens=250,
            messages=messages,
            temperature=0.5,
        )
        response_content = response.choices[0].message.content

    elif selected_llm == 'mistral-medium':
        response = client.chat.complete(
            model='mistral-medium-latest',
            max_tokens=250,
            messages=messages,
            temperature=0.5,
        )
        response_content = response.choices[0].message.content

    st.session_state['messages'].append({"role": "assistant", "content": response_content})

    with st.chat_message("assistant"):
        st.markdown(response_content)