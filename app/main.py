from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import os
import shutil

from app.embeddings import get_embeddings
from app.vectorstore import (
    load_vectorstore,
    create_vectorstore,
    save_vectorstore
)
from app.retriever import retrieve_documents, build_context
from app.rag import get_llm, generate_answer

from app.loaders import loader_func
from app.chunking import chunk_documents


app = FastAPI()


# -------------------------
# Configuration
# -------------------------

VECTORSTORE_PATH = "vector_disk"
UPLOAD_DIR = "uploads"


# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------
# Models
# -------------------------

class ChatRequest(BaseModel):
    question: str


# -------------------------
# Load existing RAG system
# -------------------------

embeddings = get_embeddings()
llm = get_llm()

vectorstore = load_vectorstore(
    VECTORSTORE_PATH,
    embeddings
)


# -------------------------
# Chat endpoint
# -------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    results = retrieve_documents(
        vectorstore,
        request.question,
        k=3
    )

    context = build_context(results)

    answer = generate_answer(
        llm,
        context,
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }


# -------------------------
# Document upload endpoint
# -------------------------

@app.post("/documents")
def upload_document(
    file: UploadFile = File(...)
):

    global vectorstore

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    # Only allow PDF for now
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Create file path
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:

        # -------------------------
        # 1. Save uploaded file
        # -------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(f"File saved: {file_path}")


        # -------------------------
        # 2. Load document
        # -------------------------

        pages = loader_func(file_path)

        print(
            f"Document loaded: {len(pages)} pages"
        )


        # -------------------------
        # 3. Chunk document
        # -------------------------

        chunks = chunk_documents(pages)

        print(
            f"Chunks created: {len(chunks)}"
        )


        # -------------------------
        # 4. Create embeddings
        # -------------------------

        document_embeddings = get_embeddings()

        print("Embedding model loaded")


        # -------------------------
        # 5. Create vector store
        # -------------------------

        new_vectorstore = create_vectorstore(
            chunks,
            document_embeddings
        )

        print("Vector store created")


        # -------------------------
        # 6. Save vector store
        # -------------------------

        save_vectorstore(
            new_vectorstore,
            VECTORSTORE_PATH
        )

        print(
            f"Vector store saved to: {VECTORSTORE_PATH}"
        )


        # -------------------------
        # 7. Update in-memory vectorstore
        # -------------------------

        vectorstore = new_vectorstore


    except Exception as e:

        print(
            f"Document processing failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


    finally:

        file.file.close()


    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }