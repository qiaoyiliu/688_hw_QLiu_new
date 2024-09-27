import streamlit as st
import requests
import json
from openai import OpenAI

st.title("Joy's Lab5 Weather App")
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
