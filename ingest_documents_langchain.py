import argparse
import logging
import os
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from local_rag import LocalRAG

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def smart_chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
    """
    Intelligently split text into chunks while trying to preserve semantic meaning.
    Uses RecursiveCharacterTextSplitter which tries to split on paragraph, then sentence, then word boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def process_pdf(file_path: str) -> List[Dict]:
    """
    Process a PDF file and return its chunks with metadata
    """
    logger.info(f"Processing PDF file: {file_path}")
    try:
        # Load PDF and get pages
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        texts = []
        metadatas = []
        
        for i, page in enumerate(pages):
            # Get chunks from the page content
            page_chunks = smart_chunk_text(page.page_content)
            
            # Add each chunk with metadata
            for j, chunk in enumerate(page_chunks):
                texts.append(chunk)
                metadatas.append({
                    "source": os.path.basename(file_path),
                    "page": page.metadata["page"],
                    "chunk": j,
                    "total_pages": len(pages)
                })
        
        return texts, metadatas
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        return [], []

def ingest_documents(directory_path: str):
    """
    Ingest documents (PDFs, txt, md) from a directory into the RAG system
    """
    rag = LocalRAG()
    texts = []
    metadatas = []

    # Check if directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")

    # Process all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if filename.endswith(".pdf"):
            # Handle PDF files
            pdf_texts, pdf_metadatas = process_pdf(file_path)
            texts.extend(pdf_texts)
            metadatas.extend(pdf_metadatas)
            
        elif filename.endswith((".txt", ".md")):
            # Handle text files
            logger.info(f"Processing text file: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Split text content into chunks
                    chunks = smart_chunk_text(content)
                    texts.extend(chunks)
                    # Add metadata for each chunk
                    for i, _ in enumerate(chunks):
                        metadatas.append({
                            "source": filename,
                            "chunk": i
                        })
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")

    if texts:
        rag.add_documents(texts, metadatas)
        logger.info(f"Successfully ingested {len(texts)} document chunks")
    else:
        logger.warning("No valid documents found to ingest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing documents to ingest",
    )

    args = parser.parse_args()
    ingest_documents(args.dir)
