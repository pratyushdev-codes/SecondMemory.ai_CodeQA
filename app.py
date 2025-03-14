from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import uuid

# Load environment variables
load_dotenv()

app = FastAPI(title="Codebase RAG API", description="API for querying GitHub codebases using RAG")

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
        print(f"Directory {target_dir} already exists. Skipping clone.")

def convert_files_to_txt(src_dir: str, dst_dir: str) -> None:
    """Convert repository files to txt format while preserving directory structure."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    for file_path in src_path.rglob('*'):
        if file_path.is_file() and not file_path.suffix == '.jpg':
            relative_path = file_path.relative_to(src_path)
            new_path = dst_path / relative_path.parent / (relative_path.name + '.txt')
            new_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                content = file_path.read_text(encoding='utf-8')
                new_path.write_text(content, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='latin-1')
                    new_path.write_text(content, encoding='utf-8')
                except UnicodeDecodeError:
                    print(f"Failed to decode file: {file_path}")

def process_documents(src_dir: str) -> List:
    """Load and process documents from the source directory."""
    loader = DirectoryLoader(
        src_dir,
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Number of files loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Number of chunks: {len(split_documents)}")

    for doc in split_documents:
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
        model="models/embedding-001"
    )

    return llm, embeddings

def create_vector_store(documents: List, embeddings, store_path: str) -> Qdrant:
    """Create and populate the vector store."""
    return Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=store_path,
        collection_name="my_documents"
    )

def create_qa_chain(llm, vector_store: Qdrant):
    """Create the question-answering chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
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
        print(f"Cleaned up directory: {base_dir}")

@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query a GitHub repository with a specific question."""
    # Create unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Set up directories
    base_dir = f"temp/{request_id}"
    repo_dir = f"{base_dir}/codebase"
    converted_dir = f"{base_dir}/converted_codebase"
    vector_store_path = f"{base_dir}/local_qdrant"
    
    try:
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Clone repository
        clone_repository(request.github_url, repo_dir)
        
        # Convert files to txt
        convert_files_to_txt(repo_dir, converted_dir)
        
        # Process documents
        documents = process_documents(converted_dir)
        
        # Initialize models
        llm, embeddings = initialize_models()
        
        # Create vector store
        vector_store = create_vector_store(documents, embeddings, vector_store_path)
        
        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_store)
        
        # Execute query
        response = qa_chain.invoke(request.query)
        
        # Format source documents
        formatted_docs = format_source_documents(response['source_documents'])
        
        # Prepare response
        result = {
            "result": response['result'],
            "source_documents": formatted_docs
        }
        
        # Clean up after processing
        cleanup_directories(base_dir)
        
        return result
        
    except Exception as e:
        # Clean up in case of error
        cleanup_directories(base_dir)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

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