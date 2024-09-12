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
#import time
#import os
#import logging
#from openai import AzureOpenAI

st.title("Joy's Document question answering for HW2")
st.write(
"Upload a document below and ask a question about it – GPT will answer! "
"To use this app, you need to provide an API key."
)

#read PDF files
def read_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

#read URL
def read_url_content(url):
     try:
        response=requests.get(url)
        response.raise_for_status() 
        soup=BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
     except requests.RequestException as e:
        print(f"Error reading {url}:{e}")
        return None


#which LLM selected from sidebar
def display(selected_llm):
    client = None

    if selected_llm == 'gpt-4o-mini':
        api_key = st.text_input("OpenAI API Key", type="password")
        #api_key = st.secrets['OPENAI_API_KEY']
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            st.warning("Please provide OpenAI API key")
            return
    elif selected_llm == 'claude-3-haiku-20240307':
        api_key = st.text_input("Anthropic API Key", type="password")
        #api_key = st.secrets['ANTHROPIC_API_KEY']
        if api_key:
            client = Anthropic(api_key=api_key)
        else:
            st.warning("Please provide Anthropic API key")
            return
    elif selected_llm == 'mistral-small-latest':
        api_key = st.text_input("Mistral API Key", type="password")
        #api_key = st.secrets['MISTRAL_API_KEY']
        if api_key:
            client = Mistral(api_key=api_key)
        else:
            st.warning("Please provide Mistral API key")
            return
    #else: 
        #st.info("Please add your OpenAI API key to continue.", icon="🗝️")

    #ask user to upload file
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, or .pdf)", type=("txt", "pdf")
    )

    #or ask user to paster URL
    question_url = st.text_area(
        "Or insert an URL:",
        placeholder="Copy URL here",
    )

    #ask user to select language
    languages = ['English', 'Spanish', 'French']
    selected_language = st.selectbox('Select your language:', languages)
    st.write(f"You have selected: {selected_language}")


    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if client is None:
        st.info("Please enter API key to continue.")
    else:
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
        
        else:
            url_content = read_url_content(question_url)
            messages = [
                {
                "role": "user",
                "content": f"Respond in {selected_language}. Here's a URL: {url_content} \n\n---\n\n {question}",
                }
            ]
        
        #if using gpt-4o-mini
        if selected_llm == "gpt-4o-mini":
            stream = client.chat.completions.create(
                model=selected_llm,
                max_tokens=250,
                messages=messages,
                stream=True,
                temperature=0.5,
            )
            
            st.write_stream(stream)
        
        elif selected_llm == 'claude-3-haiku-20240307':
            message = client.messages.create(
                model=selected_llm,
                max_tokens=256,
                messages=messages,
                temperature=0.5,
            )
            data = message.content[0].text
            st.write(data)
        
        elif selected_llm == 'mistral-small-latest':
            response = client.chat.complete(
                model=selected_llm,
                max_tokens=250,
                messages=messages,
                temperature=0.5,
            )
            data = response.choices[0].message.content
            st.write(data)
    








#openai_api_key = st.text_input("OpenAI API Key", type="password")
#openai_api_key = st.secrets["OPENAI_API_KEY"]



       






#question_to_ask = "Why are LLMs (AI) a danger to society?"
#system_message = """
#Goal: Answer the question using bullets. 
#      The answer should be appropriate for a 10 year old child to understand
#"""



def output_info(content, start_time, model_info):
   end_time = time.time()
   time_taken = end_time - start_time
   time_taken = round(time_taken * 10)/10

   output = f"For {model_info}, time taken = " + str(time_taken)
   logging.info(output)
   logging.info(f"  --> {content}")

   str.write(output)

def do_openAI(model):
   client = AzureOpenAI(
      api_version=openai.api_version,
      azure_endpoint=openai.api_base,
      api_key=openai_api_key
   )
   
   message_to_LLM = [
      {"role":"system", "content":system_message},
      {"role":"user", "content":question_to_ask}
   ]

   completion = client.chat.completions.create(
      model = "gpt-4o-mini",
      messages=message_to_LLM,
      temperature=0,
      seed=10,
      max_tokens=1500,
      stream=True
   ) 

   content = ""

