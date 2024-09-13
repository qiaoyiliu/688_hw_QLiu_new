import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup


def read_url_content(url):
     try:
        response=requests.get(url)
        response.raise_for_status() 
        soup=BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
     except requests.RequestException as e:
        print(f"Error reading {url}:{e}")
        return None
     
# Show title and description.
st.title("Joy's HW3 question answering chatbot")

llm_model = st.sidebar.selectbox("Which model?",
                                    ("gpt-4o-mini", "gpt-4o")
                                     #"claude-haiku, claude-opus",
                                     #"mistral-small", "mistral-medium")
)

if llm_model == "gpt-4o-mini":
    model_to_use = "gpt-4o-mini"
elif llm_model == "gpt-4o":
    model_to_use = "gpt_4o"
#elif llm_model == "claude-haiku":
#    model_to_use = "claude-3-haiku-20240307"
#elif llm_model == "claude-opus":
#    model_to_use = "claude-3-opus-20240229"
#elif llm_model == "mistral-small":
#    model_to_use = "mistral-small-latest"
#elif llm_model == "mistral-medium":
#    model_to_use = "mistral-medium-latest"

whether_url = st.sidebar.selectbox("Whether input 2 URLs?",
                                   ("Yes", "No"))


memory_option = st.sidebar.selectbox("Which type of LLM short-term memory?",
                                     ("Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens"))


if "client" not in st.session_state:
    #openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai_api_key = st.text_input("Insert OpenAI API Key", type="password")
    st.session_state.client = OpenAI(api_key=openai_api_key)

if whether_url == "Yes":
    question_url_1 = st.text_area("Insert first URL:", 
                                    placeholder="Copy URL here",)
    question_url_2 = st.text_area("Insert second URL (optional):", 
                                    placeholder="Copy URL here",)
else:
    question_url_1 = st.text_area("Insert first URL:", 
                                    placeholder="Copy URL here",)

#choose buffer of 5 questions------------------------------------------
if memory_option == "Buffer of 5 questions":
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "questions_buffer" not in st.session_state:
        st.session_state.questions_buffer = []
    
    def update_q_buffer(new_q):
        st.session_state["messages"].append(new_q)
        st.session_state["messages"] = st.session_state["messages"][-5:]
    
    question = read_url_content(question_url_1)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask me anything?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    update_q_buffer(prompt)

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Submit URL"):
        if question_url_1:
            url_content = read_url_content(question_url_1)
            if url_content:
                st.session_state.messages.append({"role": "user", "content": f"URL Content from {url_input}"})
                with st.chat_message("user"):
                    st.markdown(f"URL Content from {question_url_1}")
                update_q_buffer(url_content[:1000])

    if st.session_state.questions_buffer:
        st.write("Last 5 questions were:")
        for i, question in enumerate(st.session_state.questions_buffer,1):
            st.write(f"{i}. {question}")

#choose summary of conversation:
elif memory_option == "Conversation summary":
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    def summarize_conversation(messages):
        try:
            conversation_log = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages])
            client = st.session_state.client
            response = OpenAI.Completion.create(
                engine="gpt-4o-mini",
                prompt=f"Summarize the following conversation:\n{conversation_log}\nSummary:",
                max_tokens=150,
            )   
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error in summarizing: {e}"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask me anything?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = st.session_state.client
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Summarize Conversation"):
        if st.session_state.messages:
            summary = summarize_conversation(st.session_state.messages)
            st.write("Conversation Summary:")
            st.write(summary)
        else:
            st.write("No conversation history to summarize.")    

elif memory_option == "Buffer of 5,000 tokens":
    def count_tokens(text):
        return len(text.split())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    
    def update_chat_history_and_token_count(role, content):
        new_tokens = count_tokens(content)
        st.session_state.messages.append({"role": role, "content": content})
        st.session_state.token_count += new_tokens
        while st.session_state.token_count > 5000:
            oldest_message = st.session_state.messages.pop(0)
            st.session_state.token_count -= count_tokens(oldest_message["content"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask me anything?"):
        update_chat_history_and_token_count("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)        

        client = st.session_state.client
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        update_chat_history_and_token_count("assistant", response)

        st.write(f"Total tokens used in the conversation: {st.session_state.token_count} / 5000")

