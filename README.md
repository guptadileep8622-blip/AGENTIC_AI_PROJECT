# Basic RAG System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content through a FastAPI backend.

The system retrieves relevant document chunks and uses a large language model (LLM) to generate answers based on the retrieved context.

## Features

- PDF document loading using PyPDFLoader
- Text chunking using RecursiveCharacterTextSplitter
- Local embeddings using Ollama and `nomic-embed-text`
- Vector storage and similarity search using FAISS
- Retrieval of the top-k relevant document chunks
- Answer generation using Groq-hosted LLMs
- Persistent vector storage on disk
- FastAPI endpoints for document upload and question answering
- Logging and basic exception handling
- Interactive API testing through Swagger UI

## Tech Stack

- Python
- FastAPI
- LangChain
- Ollama
- Groq
- FAISS
- PyPDF
- Uvicorn

## Project Structure

```text
AGENTIC_AI_PROJECT/
├── app/
│   ├── main.py
│   ├── logger.py
│   ├── config.py
│   ├── loaders.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── rag.py
├── documents/
├── uploads/
├── vector_disk/
├── index.py
├── run_rag.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## How It Works

1. A PDF document is loaded and its text is extracted.
2. The extracted text is divided into smaller, overlapping chunks.
3. An embedding model converts the chunks into numerical vectors.
4. FAISS stores the vectors for similarity search.
5. A user submits a question through the API.
6. The system retrieves the most relevant chunks.
7. The retrieved context and question are passed to the LLM.
8. The LLM generates an answer based on the retrieved context.

## Prerequisites

- Python and pip
- Ollama installed and running
- The `nomic-embed-text` embedding model downloaded
- A valid Groq API key

Download the embedding model if it is not already installed:

```bash
ollama pull nomic-embed-text
```

## Installation

Clone the repository and navigate to the project directory.

Create and activate your Python environment if you do not already have one:

```bash
conda create -n basic-rag python=3.12
conda activate basic-rag
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables in `.env`. For example:

```text
GROQ_API_KEY=your_groq_api_key
```

Keep your actual API key private. Never commit `.env` to Git.

## Running the Project

### 1. Build the vector index

If you are indexing the initial documents, place them in the `documents/` directory and run:

```bash
python index.py
```

This loads the configured document, creates chunks and embeddings, and saves the FAISS index to `vector_disk/`.

Run this step again whenever you need to rebuild the index using the documents configured in the indexing script.

### 2. Start the API server

Make sure the persisted vector index exists, then run:

```bash
uvicorn app.main:app --reload
```

### 3. Open Swagger UI

Visit:

http://127.0.0.1:8000/docs

Use the interactive interface to test the endpoints.

## API Endpoints

### POST `/documents`

Uploads and indexes a PDF document.

The endpoint saves the PDF, extracts its content, creates chunks and embeddings, and persists the resulting FAISS index.

### POST `/chat`

Accepts a question and returns an answer generated using retrieved document context.

Example request:

```json
{
  "question": "What is GraphRAG?"
}
```

The response contains the submitted question and the generated answer.

## Current Limitations

- Uploading a new PDF replaces the existing FAISS index rather than incrementally adding to it.
- Only PDF uploads are supported by the upload endpoint.
- The application is a learning prototype, not a production-hardened service.
- Authentication, automated testing, upload-size limits, and robust concurrent ingestion are not yet implemented.
- Answer quality depends on document extraction, chunking, retrieval, and the selected LLM.

## Future Improvements

- Incremental indexing and multi-document management
- Improved retrieval and reranking
- Automated evaluation of retrieval and answer quality
- Authentication and stronger input validation
- Background processing for large documents
- Automated tests and CI/CD
- Deployment and observability

## Project Goal

This project is the first milestone in a larger Agentic RAG System learning project. The goal is to progressively build a more reliable system with production-oriented retrieval, agent routing, live integrations, memory, guardrails, evaluation, observability, and deployment.
