import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("Joy's Lab3 question answering chatbot")

openAI_model = st.sidebar.selectbox("Which model?",
                                    ("mini", "regular"))
if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt_4o"

if "client" not in st.session_state:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    #openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.session_state.client = OpenAI(api_key=openai_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = \
        [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("What is up?"):
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
