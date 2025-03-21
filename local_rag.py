import argparse
import logging
import os
from typing import Dict, List, Optional

import chromadb
import requests
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("LocalRAG")


class LocalRAG:
    def __init__(self, collection_name: str = "documents"):
        # Initialize ChromaDB as the vector store with persistent storage
        # This allows embeddings to be saved to disk at ./chroma_db rather than in-memory
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Either retrieve existing collection or create new one
        # Collections in ChromaDB are namespaced groups of embeddings
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Found existing collection: {collection_name}")
        except Exception as e:
            try:
                # Create new collection with cosine similarity metric
                # HNSW is the approximate nearest neighbor algorithm used
                self.collection = self.chroma_client.create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {collection_name}")
            except Exception as create_error:
                logger.error(f"Failed to create collection: {str(create_error)}")
                raise

        # Load the embedding model - all-MiniLM-L6-v2 is a lightweight but effective
        # sentence transformer that creates 384-dimensional embeddings
        # It's optimized for semantic search
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Configure Ollama endpoint for the LLM component
        # Ollama runs locally and exposes this API endpoint
        self.ollama_url = "http://localhost:11434/api/generate"

        # System prompt that guides the LLM's behavior
        # Instructs it to use retrieved context and maintain conversation history
        # Also ensures source attribution and graceful handling of out-of-context queries
        self.system_prompt = """You are a helpful AI assistant engaging in a conversation. 
        Using the provided context and chat history, answer the user's question. 
        If you cannot answer the question based on the context, say so. 
        Always cite your sources when using information from the context."""

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store

        This method handles the document ingestion pipeline:
        1. Generates dense vector embeddings for each text using all-MiniLM-L6-v2
        2. Creates sequential document IDs (doc_0, doc_1, etc.)
        3. Stores the document-embedding pairs in ChromaDB along with optional metadata

        The embeddings are 384-dimensional vectors optimized for semantic similarity.
        ChromaDB handles the vector indexing using HNSW algorithm with cosine similarity.

        Args:
            texts: List of document texts to embed and store
            metadatas: Optional metadata for each document (e.g., source, date)
                      Defaults to empty dicts if not provided
        """
        try:
            # Generate dense embeddings - returns numpy array which we convert to list
            embeddings = self.embedding_model.encode(texts).tolist()

            # Create sequential IDs for ChromaDB's internal tracking
            ids = [f"doc_{i}" for i in range(len(texts))]

            # Store document-embedding pairs with metadata in ChromaDB
            # ChromaDB will build/update its HNSW index automatically
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=metadatas if metadatas else [{}] * len(texts),
            )
            logger.info(f"Successfully added {len(texts)} documents to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def get_relevant_context(self, query: str, n_results: int = 6) -> List[Dict]:
        """Get relevant documents for a query"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            contexts = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                contexts.append(
                    {"text": doc, "metadata": metadata, "relevance_score": 1 - distance}
                )

            return contexts

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {str(e)}")
            raise

    def get_llm_response(self, prompt: str) -> str:
        """Get response from local LLM using Ollama"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": "mistral", "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]

        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise

    def get_rag_response(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Get RAG response for a query, considering chat history"""
        try:
            # Get relevant context
            contexts = self.get_relevant_context(query)

            # Format context for prompt
            context_texts = [
                f"Source {i + 1}: {c['text']}" for i, c in enumerate(contexts)
            ]
            formatted_context = "\n\n".join(context_texts)

            # Format chat history if provided
            chat_history_text = ""
            if chat_history:
                history_messages = []
                for msg in chat_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_messages.append(f"{role}: {content}")
                chat_history_text = "\n".join(history_messages)

            # Construct prompt
            chat_history_section = (
                "Chat History:\n" + chat_history_text if chat_history_text else ""
            )
            prompt = f"""System: {self.system_prompt}

Context:
{formatted_context}

{chat_history_section}

User Question: {query}"""

            # Get response from LLM
            response = self.get_llm_response(prompt)

            return {"answer": response, "sources": [c["metadata"] for c in contexts]}

        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument(
        "--query", type=str, required=True, help="Question to ask the RAG system"
    )

    args = parser.parse_args()

    rag = LocalRAG()
    try:
        response = rag.get_rag_response(args.query)
        logger.info(f"\nQuestion: {args.query}")
        logger.info(f"Answer: {response['answer']}")
        logger.info(f"Sources: {response['sources']}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
