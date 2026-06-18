 
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
