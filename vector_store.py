import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import config

def get_chroma_client_and_collection():
    """
    Initializes a persistent Chroma client and returns (client, collection).
    """
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    return client, collection

def initialize_vector_store():
    """
    Creates and returns a StorageContext backed by ChromaVectorStore.
    """
    _, collection = get_chroma_client_and_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def get_indexed_filenames() -> list:
    """
    Queries ChromaDB to find the list of unique file names currently indexed.
    """
    try:
        _, collection = get_chroma_client_and_collection()
        results = collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []
        
        filenames = set()
        for meta in results["metadatas"]:
            # LlamaIndex maps metadata dictionary keys directly to Chroma metadata.
            if meta and "file_name" in meta:
                filenames.add(meta["file_name"])
        return sorted(list(filenames))
    except Exception as e:
        print(f"Error reading indexed filenames: {e}")
        return []

def clear_vector_store():
    """
    Deletes the current Chroma collection and recreates it.
    """
    client, _ = get_chroma_client_and_collection()
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Collection not deleted (it might not exist): {e}")
    # Recreate
    client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
