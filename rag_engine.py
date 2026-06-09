import os
import torch
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import config
from vector_store import get_chroma_client_and_collection

def initialize_llm_and_embeddings():
    """
    Initializes Llama LLM and Embedding model, setting them globally in LlamaIndex.
    """
    # 1. Device determination
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing RAG engine on device: {device}")
    
    # 2. Embedding Model Setup
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME, 
        trust_remote_code=True,
        device=device
    )
    
    # 3. LLM Setup
    # meta-llama/Llama-3.2-1B-Instruct is gated, uses HF_TOKEN from config.py / env
    llm = HuggingFaceLLM(
        context_window=config.CONTEXT_WINDOW,
        max_new_tokens=config.MAX_NEW_TOKENS,
        generate_kwargs={"temperature": config.TEMPERATURE, "do_sample": True},
        tokenizer_name=config.MODEL_NAME,
        model_name=config.MODEL_NAME,
        device_map="auto" if device == "cuda" else None,
    )
    
    # 4. Global Settings Configuration
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

def get_index(documents=None, storage_context=None):
    """
    Loads index from existing vector store if documents is None,
    otherwise creates a new index from documents.
    """
    _, collection = get_chroma_client_and_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    if documents is not None and len(documents) > 0:
        # Create from documents
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
    else:
        # Load from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model
        )
    return index

def get_query_engine(index):
    """
    Constructs a query engine with a clean custom QA prompt.
    """
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above, answer the query in a crisp and concise manner.\n"
        "If you do not know the answer based on the context, state 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    query_engine = index.as_query_engine()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
    return query_engine

def query_and_extract_sources(query_engine, query_str: str) -> tuple:
    """
    Queries the index and returns (response_text, sources_list).
    """
    response = query_engine.query(query_str)
    
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for source_node in response.source_nodes:
            node = source_node.node
            metadata = node.metadata if hasattr(node, "metadata") else {}
            
            file_name = metadata.get("file_name", "Unknown File")
            page_label = metadata.get("page_label", None) or metadata.get("page", "N/A")
            snippet = node.get_content()[:250] + "..." if hasattr(node, "get_content") else ""
            
            # Avoid duplicating citations from the same file/page
            citation_exists = any(
                s["file_name"] == file_name and s["page"] == page_label 
                for s in sources
            )
            if not citation_exists:
                sources.append({
                    "file_name": file_name,
                    "page": page_label,
                    "snippet": snippet
                })
                
    return str(response), sources
