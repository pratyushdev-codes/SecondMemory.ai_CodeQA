import os
from typing import Tuple
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Define the persist directory and collection name.
PERSIST_DIRECTORY = ".local_qdrant"
COLLECTION_NAME = "my_documents"

GITHUB_CODEBASE = ".codebase"
CONVERTED_CODEBASE = ".converted_codebase"

def delete_vector_collection(qdrant_client: QdrantClient, collection_name: str) -> Tuple[bool, str]:
    """
    Deletes an existing collection from Qdrant.
    Returns (True, message) if deletion was successful; otherwise, (False, error message).
    """
    try:
        qdrant_client.delete_collection(collection_name)
        return True, f"Collection '{collection_name}' deleted successfully."
    except Exception as e:
        return False, f"Unable to delete collection: {e}"

def delete_codebase() -> Tuple[bool, str]:
    """
    Deletes the codebase directory and its contents.
    """
    try:
        os.system(f"rm -rf {GITHUB_CODEBASE}")
        return True, "Codebase directory deleted successfully."
    except Exception as e:
        return False, f"Unable to delete codebase directory: {e}"

def delete_converted_codebase() -> Tuple[bool, str]:
    """
    Deletes the converted codebase directory and its contents.
    """
    try:
        os.system(f"rm -rf {CONVERTED_CODEBASE}")
        return True, "Converted codebase directory deleted successfully."
    except Exception as e:
        return False, f"Unable to delete converted codebase directory: {e}"
