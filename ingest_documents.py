import argparse
import logging
import os

from local_rag import LocalRAG

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_documents(directory_path: str):
    """
    Ingest all text documents from a directory into the RAG system
    """
    rag = LocalRAG()
    texts = []
    metadatas = []

    # Check if directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")

    # Process all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing file: {filename}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    metadatas.append({"source": filename})
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")

    if texts:
        rag.add_documents(texts, metadatas)
        logger.info(f"Successfully ingested {len(texts)} documents")
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
