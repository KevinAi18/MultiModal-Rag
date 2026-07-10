# Multi-PDF RAG Assistant

A local, high-performance Retrieval-Augmented Generation (RAG) assistant that allows users to upload multiple PDF documents and perform conversational Q&A. The application utilizes the open-source **LLaMA 3.2 1B Instruct** model, **LlamaIndex** for query orchestration, and **ChromaDB** as a vector database for semantic search.

This application includes a premium Streamlit web interface, citation tracking (identifying the source document and page number for each generated response), and persistent session chat history.

## Features
- **Conversational Chat UI**: A sleek, user-friendly chat interface for interacting with your documents.
- **Multi-PDF Uploads**: Dynamically upload and index multiple PDFs directly from the web interface.
- **Source Citation**: Every response lists the source document and page number(s) from which the context was retrieved.
- **Vector DB Persistence**: Re-uses already-indexed files from ChromaDB, so you don't need to re-upload documents on restart.
- **Full Clear/Reset**: Delete all cached files and collection indexes with a single click.

## System Architecture

```
├── app.py (Streamlit Web UI)
├── rag_engine.py (LLaMA model setup and LlamaIndex orchestration)
├── document_loader.py (PDF saving and loading using SimpleDirectoryReader)
├── vector_store.py (ChromaDB persistent vector store operations)
├── config.py (Hyperparameters and paths)
├── requirements.txt (Dependencies list)
└── .env (Environment variables for keys/tokens)
```

## System Requirements
- **Python**: Version 3.8 or higher.
- **Hardware**: A CUDA-compatible GPU with at least 8GB VRAM is recommended for local acceleration, and 16GB RAM of system memory.
- **Hugging Face Account**: A Hugging Face account and an API Token are required to download the gated `meta-llama/Llama-3.2-1B-Instruct` model.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/KevinAi18/MultiModal-Rag
cd MultiModal-Rag
```

### 2. Install Dependencies
Initialize a virtual environment and install the required dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root of the project directory based on the template:
```bash
copy .env.example .env
```
Open the `.env` file and insert your Hugging Face API Token:
```env
HF_TOKEN=your_huggingface_token_here
```

### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```

Open the local URL displayed in the terminal (typically `http://localhost:8501`) in your browser.

## Customization
Hyperparameters like `temperature`, `max_new_tokens`, `chunk_size`, and system directories can be customized directly in the `config.py` file or via env vars.

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
Created by [KevinAi18](https://github.com/KevinAi18).
 
## Supported Modalities 
- Text documents and PDFs 
- Images including charts, diagrams and photos 
- Audio transcripts via Whisper 
- Video keyframes with temporal alignment 
 
All modalities are indexed into a unified vector store for cross modal retrieval. 
 
## Project Structure 
- ingestion - handles parsing of text, image, audio and video inputs 
- embeddings - generates and stores unified multimodal embeddings 
- retrieval - performs cross modal search and reranking 
- generation - produces final answer using retrieved multimodal context 
 
## Limitations 
- Cross modal alignment quality depends heavily on training data 
- Video processing is computationally expensive at scale 
- Table and chart extraction accuracy varies with image quality 
 
## Contributing 
- Fork the repository and create a new feature branch 
- Keep changes focused and well documented 
- Open a pull request with screenshots or examples where relevant 
- Bug reports and suggestions welcome via GitHub Issues 
 
## Setup 
1. Clone the repository 
2. Install dependencies from requirements.txt 
3. Add API keys for your chosen LLM provider to .env file 
4. Run the ingestion script to index your documents 
5. Start the FastAPI server and query via the REST API 
 
## License 
This project is released under the MIT License. See the LICENSE file for details. 
 
## Acknowledgements 
Built using CLIP for embeddings and LangChain for retrieval orchestration. 
 
## FAQ 
Q: Can this handle very large PDF files? 
A: Yes, though very large files increase ingestion and indexing time. 
