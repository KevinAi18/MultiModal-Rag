import streamlit as st
import os
import config
import document_loader
import vector_store
import rag_engine

# 1. Page Config
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI styling
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .chat-citation-box {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 10px 15px;
        margin-top: 10px;
        border: 1px solid #334155;
    }
    .citation-title {
        font-weight: bold;
        color: #38BDF8;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .citation-snippet {
        font-size: 0.85em;
        color: #94A3B8;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-PDF RAG Assistant")
st.caption("Powered by LLaMA 3.2 1B, LlamaIndex, and ChromaDB")

# 2. Sidebar Configuration & Token Check
st.sidebar.title("Configuration & Upload")

# Hugging Face Token input (dynamic fallback if not in .env)
hf_token = os.getenv("HF_TOKEN", "")
if not hf_token:
    hf_token = st.sidebar.text_input(
        "Enter Hugging Face API Token:",
        type="password",
        help="Llama 3.2 is a gated model. Provide a token with access permissions to download it."
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        config.HF_TOKEN = hf_token
else:
    st.sidebar.success("HuggingFace Token loaded from environment.")

# Sidebar status
status_placeholder = st.sidebar.empty()

# Initialize RAG System when Token is ready
if not hf_token:
    status_placeholder.warning("⚠️ Please provide a HuggingFace API Token to initialize the application.")
    st.info("👈 Enter your Hugging Face API Token in the sidebar to download and run the Llama-3.2-1B model.")
    st.stop()

# Cache initialization of heavy models
@st.cache_resource(show_spinner="Downloading and initializing LLaMA 3.2 model (this may take a few minutes)...")
def get_rag_context():
    rag_engine.initialize_llm_and_embeddings()
    storage_context = vector_store.initialize_vector_store()
    return storage_context

try:
    storage_context = get_rag_context()
    status_placeholder.success("⚡ Model and Vector Store Initialized!")
except Exception as e:
    status_placeholder.error(f"❌ Initialization failed: {e}")
    st.error(f"Error loading model: {e}. Ensure your Hugging Face Token has permission for `meta-llama/Llama-3.2-1B-Instruct`.")
    st.stop()

# 3. Handle PDF Uploads
st.sidebar.subheader("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("Process & Index Documents", use_container_width=True)

# 4. Database Reset
clear_clicked = st.sidebar.button("Clear Indexed Database", use_container_width=True, type="secondary")

if clear_clicked:
    with st.spinner("Clearing database..."):
        vector_store.clear_vector_store()
        document_loader.clear_upload_dir()
        if "query_engine" in st.session_state:
            del st.session_state["query_engine"]
        if "index" in st.session_state:
            del st.session_state["index"]
        st.session_state.messages = []
        st.sidebar.success("Database cleared!")
        st.rerun()

# 5. Load or Update Index
if process_clicked and uploaded_files:
    with st.spinner("Processing documents (saving, chunking, and embedding)..."):
        # Save files to UPLOAD_DIR
        document_loader.save_uploaded_files(uploaded_files)
        # Load and parse
        docs = document_loader.load_documents_from_dir()
        if docs:
            # Recreate or build index
            index = rag_engine.get_index(documents=docs, storage_context=storage_context)
            st.session_state["index"] = index
            st.session_state["query_engine"] = rag_engine.get_query_engine(index)
            st.sidebar.success(f"Successfully indexed {len(uploaded_files)} PDF(s)!")
            st.rerun()
        else:
            st.sidebar.error("Failed to load documents.")

# Load active index from vector store on startup
if "query_engine" not in st.session_state:
    indexed_files = vector_store.get_indexed_filenames()
    if indexed_files:
        # Load existing collection index
        index = rag_engine.get_index(documents=None, storage_context=storage_context)
        st.session_state["index"] = index
        st.session_state["query_engine"] = rag_engine.get_query_engine(index)

# 6. Sidebar display of currently indexed documents
st.sidebar.subheader("Currently Indexed Documents")
indexed_files = vector_store.get_indexed_filenames()
if indexed_files:
    for filename in indexed_files:
        st.sidebar.markdown(f"📄 `{filename}`")
else:
    st.sidebar.info("No documents currently indexed.")

# 7. Chat Interface & Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"**Document**: `{src['file_name']}` | **Page**: `{src['page']}`")
                    st.markdown(f"<div class='chat-citation-box'><div class='citation-title'>Context snippet:</div><div class='citation-snippet'>\"{src['snippet']}\"</div></div>", unsafe_allow_html=True)

# Prompt input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process using RAG engine
    if "query_engine" in st.session_state:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Searching and generating answer...")
            
            try:
                response_text, sources = rag_engine.query_and_extract_sources(
                    st.session_state["query_engine"], 
                    prompt
                )
                
                message_placeholder.markdown(response_text)
                
                # Show sources
                if sources:
                    with st.expander("📚 View Sources"):
                        for src in sources:
                            st.markdown(f"**Document**: `{src['file_name']}` | **Page**: `{src['page']}`")
                            st.markdown(f"<div class='chat-citation-box'><div class='citation-title'>Context snippet:</div><div class='citation-snippet'>\"{src['snippet']}\"</div></div>", unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error querying response: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload and index documents before asking questions, or enter your Hugging Face Token in the sidebar.")
