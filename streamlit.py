import streamlit as st
import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="Codebase RAG Explorer",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("Codebase RAG Explorer")
st.markdown("""
This app allows you to analyze GitHub repositories using Retrieval Augmented Generation (RAG).
Simply enter a GitHub repository URL and ask questions about the codebase.
""")

# API Configuration
api_url = st.sidebar.text_input("API URL", value="http://0.0.0.0:8000/query")

with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This application connects to a FastAPI backend that:
    1. Clones the specified GitHub repository
    2. Processes code files into text chunks
    3. Creates embeddings using Google Generative AI
    4. Uses RAG with Gemini 1.5 Flash to answer your questions
    """)

# Main input form
with st.form("repository_form"):
    # Input for GitHub repository URL
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository"
    )
    
    # Input for query
    query = st.text_area(
        "Your Question",
        placeholder="How is authentication implemented in this codebase?",
        height=100
    )
    
    # Submit button
    submit_button = st.form_submit_button("Analyze Repository")

# Process the request when submitted
if submit_button:
    if not repo_url or not query:
        st.error("Please provide both a GitHub repository URL and a question.")
    else:
        # Show loading state
        with st.spinner("Processing repository... This may take a few minutes"):
            try:
                # Prepare the request data
                payload = {
                    "github_url": repo_url,
                    "query": query
                }
                
                # Make API request
                response = requests.post(
                    f"{api_url}/query",
                    json=payload,
                    timeout=300  # 5-minute timeout
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display the answer
                    st.success("Analysis complete!")
                    st.subheader("Answer")
                    st.markdown(data["result"])
                    
                    # Display source documents
                    st.subheader("Source Code References")
                    for i, doc in enumerate(data["source_documents"]):
                        with st.expander(f"Source {i+1}: {doc['source']}"):
                            st.code(doc["content"], language="python")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("Request timed out. The repository might be too large or the server is busy.")
            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to the API server at {api_url}. Please make sure the server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display sample questions
st.subheader("Sample Questions")
sample_questions = [
    "What authentication methods are used in this codebase?",
    "Explain the main components of this application and how they interact.",
    "How does the error handling work in this codebase?",
    "What database models are defined and what are their relationships?",
    "How are API endpoints secured in this application?"
]

for q in sample_questions:
    if st.button(q):
        # Fill the form with the sample question
        st.session_state["query"] = q
        # Rerun to update the text area
        st.experimental_rerun()

# Session state to store history
if "history" not in st.session_state:
    st.session_state.history = []

# Display history if available
if st.session_state.history:
    st.subheader("Previous Queries")
    for item in st.session_state.history:
        with st.expander(f"Q: {item['query'][:50]}..."):
            st.markdown(f"**Repository:** {item['repo_url']}")
            st.markdown(f"**Question:** {item['query']}")
            st.markdown(f"**Answer:** {item['answer']}")