 
## 2026-06-13 
### Multimodal RAG - CLIP Embeddings 
- CLIP model jointly embeds images and text in same space 
- Used for image-text similarity search in multimodal RAG 
- LLaVA and GPT-4V used as multimodal answer generators 
- Challenge: aligning image and text embedding dimensions 
 
## 2026-06-14 
### Image Captioning in Multimodal RAG 
- Image captioning converts visual content into text for retrieval 
- BLIP-2 and LLaVA used for generating image captions 
- Captions stored alongside image embeddings in vector DB 
- Hybrid search over captions and raw embeddings improves recall 
 
## 2026-06-16 
### ColPali Visual Retrieval Notes 
- ColPali retrieves documents directly from page images 
- No need for OCR or text extraction from PDFs 
- PaliGemma vision model used as backbone for embeddings 
- Late interaction scoring similar to ColBERT for text retrieval 
 
## 2026-06-19 
### Vision Transformer in Multimodal RAG 
- Vision Transformer splits image into fixed size patches 
- Each patch is embedded as token similar to text tokens 
- ViT embeddings used for image retrieval in multimodal pipeline 
- Combining ViT with text encoder improves cross modal search 
 
## 2026-06-21 
### Table Extraction in Multimodal RAG 
- Tables in PDFs are hard to parse with standard text extractors 
- Unstructured library detects and extracts tables from documents 
- Table Transformer model used for table detection in images 
- Extracted tables converted to markdown for LLM understanding 
 
## 2026-06-23 
### Audio Integration in Multimodal RAG 
- Audio modality adds speech and sound understanding to RAG 
- Whisper transcribes audio to text for retrieval pipeline 
- Audio embeddings from wav2vec used for direct audio search 
- Multimodal index stores text image and audio in unified store 
 
## 2026-06-26 
### Cross Modal Attention Notes 
- Improves grounding of generated text in visual content 
- Used in models like Flamingo and BLIP-2 for vision language tasks 
- Attention maps help visualize which image regions model focuses on 
 
## 2026-06-28 
### Chart and Graph Understanding Notes 
- Charts require specialized vision models beyond plain OCR 
- DePlot converts chart images into structured table format 
- MatCha model trained specifically for chart question answering 
- Extracted chart data combined with text for full context retrieval 
 
## 2026-06-30 
### Video Understanding in Multimodal RAG 
- Video frames sampled at intervals for keyframe based retrieval 
- Frame embeddings combined with audio transcript for full context 
- Video-LLaVA model used for direct video question answering 
- Temporal alignment matches transcript timestamps to video frames 
 
## 2026-07-03 
### Multimodal Embedding Storage Notes 
- Separate vector indices used for text image and audio embeddings 
- Unified embedding space allows cross modal similarity search 
- Weaviate and Qdrant both support multi vector storage natively 
- Metadata filtering combined with vector search for precise results 
 
## 2026-07-05 
### Multimodal Reranking Notes 
- Reranking in multimodal RAG scores image and text results jointly 
- CLIP score used as initial filter before cross encoder reranking 
- Late fusion combines separate text and image relevance scores 
- Helps surface most relevant mixed media result to the user 
