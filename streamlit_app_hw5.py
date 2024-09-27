import streamlit as st
import requests
import json
from openai import OpenAI

st.title("Joy's HW5 Using functions to answer course-related questions")

def relevant_course_info(query: str) -> str:
    """
    Function to query ChromaDB using a query embedding and return relevant course content.
    """
    try:
        openai_client = st.session_state.openai_client
        
        # Generate embeddings for the user query
        query_response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = query_response.data[0].embedding

        # Search for the top 3 relevant documents in the ChromaDB
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
            return "No relevant course information found."

        # Combine all relevant course information
        context = "\n\n".join(relevant_documents)
        return context
    
    except Exception as e:
        return f"Query failed with error: {e}"

# Define tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_course_info",
            "description": "Retrieve relevant course information from the uploaded syllabus",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or query about the course content.",
                    }
                },
                "required": ["query"],
            }
        }
    }
]

# User's question or prompt
messages = [
    {"role": "user", "content": "What are the main topics covered in the syllabus?"}
]

# Step 1: Send the initial message to the model
response = openai_client.chat.completions.create(
    model='gpt-4o',
    messages=messages,
    tools=tools,
    function_call="auto"  # Let the model decide if it needs to call a function
)

# Append the response to the message list
response_message = response.choices[0].message
messages.append(response_message)

# Step 2: Check if the model includes a tool call
if response_message.get("function_call"):
    tool_function_name = response_message["function_call"]["name"]
    tool_query_string = json.loads(response_message["function_call"]["arguments"])["query"]

    # Step 3: Call the relevant function and retrieve results
    if tool_function_name == "relevant_course_info":
        results = relevant_course_info(tool_query_string)
        
        # Append function results to messages list
        messages.append({
            "role": "tool", 
            "name": tool_function_name, 
            "content": results
        })
        
        # Step 4: Get a new response from the model based on the function's output
        model_response_with_function_call = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Print the final model response after the function call
        print(model_response_with_function_call.choices[0].message["content"])
    
    else:
        print(f"Error: function {tool_function_name} does not exist")
else:
    # If no tool was called, return the assistant's regular response
    print(response_message["content"])
