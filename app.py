import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
import numpy as np

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow CORS for all origins
app = FastAPI(title="Codebase RAG API", description="API for querying GitHub codebases using RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class QueryRequest(BaseModel):
    github_url: str
    query: str

class QueryResponse(BaseModel):
    result: str
    source_documents: List[Dict[str, Any]]

def clone_repository(repo_url: str, target_dir: str) -> None:
    """Clone a git repository to the specified directory."""
    if not os.path.exists(target_dir):
        os.system(f"git clone {repo_url} {target_dir}")
    else:
        logger.info(f"Directory {target_dir} already exists. Skipping clone.")

def convert_files_to_txt(src_dir: str, dst_dir: str) -> None:
    """Convert repository files to txt format while preserving directory structure."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    excluded_extensions = {".jpg", ".png", ".pdf", ".zip", ".exe", ".bin"}
    excluded_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "public"}
    
    for file_path in src_path.rglob('*'):
        # Skip excluded directories
        if any(excluded_dir in str(file_path) for excluded_dir in excluded_dirs):
            continue
            
        if file_path.is_file() and file_path.suffix not in excluded_extensions:
            relative_path = file_path.relative_to(src_path)
            new_path = dst_path / relative_path.parent / (relative_path.name + ".txt")
            new_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                content = file_path.read_text(encoding="utf-8")
                new_path.write_text(content, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding="latin-1")
                    new_path.write_text(content, encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode file: {file_path}")

def process_documents(src_dir: str) -> List:
    """Load and process documents from the source directory."""
    loader = DirectoryLoader(src_dir, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    logger.info(f"Number of files loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size for better embedding performance
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"Number of chunks: {len(split_documents)}")

    for doc in split_documents:
        if "source" in doc.metadata:
            doc.metadata["source"] = doc.metadata["source"].replace(".txt", "")

    return split_documents

def initialize_models():
    """Initialize the LLM and embeddings models."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT"  # Specify task type explicitly
    )

    return llm, embeddings

def create_vector_store(documents: List, embeddings_model, store_path: str):
    """Create and populate the FAISS vector store."""
    try:
        logger.info("Creating FAISS vector store from documents")
        # Create a FAISS vector store directly from the documents using the embeddings model.
        vector_store = FAISS.from_documents(documents, embeddings_model)
        
        # Persist the FAISS index to disk.
        vector_store.save_local(store_path)
        logger.info(f"Vector store created successfully with {len(documents)} documents")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception(f"Failed to create vector store: {str(e)}")

def create_qa_chain(llm, vector_store):
    """Create the question-answering chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Increased k for better context
        return_source_documents=True
    )

def format_source_documents(source_docs):
    """Format source documents for API response."""
    formatted_docs = []
    for doc in source_docs:
        formatted_docs.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content
        })
    return formatted_docs

def cleanup_directories(base_dir: str):
    """Clean up temporary directories after processing."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        logger.info(f"Cleaned up directory: {base_dir}")

@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query a GitHub repository with a specific question."""
    request_id = str(uuid.uuid4())
    base_dir = f"temp/{request_id}"
    repo_dir = f"{base_dir}/codebase"
    converted_dir = f"{base_dir}/converted_codebase"
    # Updated vector store directory name to reflect FAISS usage
    vector_store_path = f"{base_dir}/faiss_index"
    
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        # Clone and process repo
        logger.info(f"Cloning repository: {request.github_url}")
        clone_repository(request.github_url, repo_dir)
        
        logger.info("Converting files to txt format")
        convert_files_to_txt(repo_dir, converted_dir)
        
        logger.info("Processing documents")
        documents = process_documents(converted_dir)
        
        if not documents:
            cleanup_directories(base_dir)
            raise HTTPException(status_code=400, detail="No valid documents found in the repository.")

        # Initialize models
        logger.info("Initializing models")
        llm, embeddings = initialize_models()
        
        # Create vector store using FAISS
        logger.info("Creating vector store")
        vector_store = create_vector_store(documents, embeddings, vector_store_path)
        
        # Create QA chain
        logger.info("Creating QA chain")
        qa_chain = create_qa_chain(llm, vector_store)
        
        # Execute query
        logger.info(f"Executing query: {request.query}")
        response = qa_chain.invoke(request.query)
        
        # Format source documents
        formatted_docs = format_source_documents(response.get("source_documents", []))
        
        # Prepare response
        result = {
            "result": response.get("result", "No response generated"),
            "source_documents": formatted_docs
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        # Clean up after processing
        cleanup_directories(base_dir)

@app.get("/")
async def root():
    """API welcome endpoint."""
    return {
        "message": "Welcome to the Codebase RAG API",
        "endpoints": {
            "/query": "POST - Query a GitHub repository with 'github_url' and 'query' parameters"
        },
        "usage": "Send a POST request to /query with JSON body containing 'github_url' and 'query' fields"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
