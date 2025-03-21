# Local RAG System

## Setup

I am using python==3.11.11 and mistral:latest that has 4.1 GB size.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install Ollama (https://ollama.ai/) and pull the Mistral model:
```bash
ollama pull mistral
```

3. Create a data directory and add your documents:
```bash
mkdir data
# Add your .txt or .md files to the data directory
```

4. Ingest your documents:
```bash
python ingest_documents.py
```

## Usage

### CLI Mode
```bash
python cli.py --query "Your question here"
```

### Interactive CLI Mode
```bash
python cli.py
```

### API Server
```bash
uvicorn api:app --reload
```
Then access the API at http://localhost:8000

## Project Structure
- `local_rag.py`: Main RAG implementation
- `ingest_documents.py`: Document ingestion script
- `api.py`: FastAPI server
- `cli.py`: Command-line interface
- `data/`: Directory for your documents
- `chroma_db/`: Directory where ChromaDB stores embeddings
