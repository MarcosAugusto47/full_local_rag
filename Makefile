clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache

ingest:
	python ingest_documents.py --dir data

pdf_ingest:
	python ingest_documents_langchain.py --dir data_pdf

chat:
	python chatbot.py

