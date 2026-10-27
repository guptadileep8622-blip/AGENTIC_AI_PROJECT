from app.loaders import loader_func
from app.chunking import chunk_documents
from app.embeddings import get_embeddings
from app.vectorstore import create_vectorstore, save_vectorstore


PDF_PATH = "/Users/apple/Downloads/AGENTIC_AI_PROJECT/documents/electronics-14-02102-v2.pdf"
VECTORSTORE_PATH = "vector_disk"


try:
    # 1. Load document
    pages = loader_func(PDF_PATH)
    print(f"Document loaded: {len(pages)} pages")

    # 2. Chunk document
    chunks = chunk_documents(pages)
    print(f"Chunks created: {len(chunks)}")

    # 3. Get embedding model
    embeddings = get_embeddings()
    print("Embedding model loaded")

    # 4. Create vector store
    vectorstore = create_vectorstore(chunks, embeddings)
    print("Vector store created")

    # 5. Save vector store
    save_vectorstore(vectorstore, VECTORSTORE_PATH)
    print(f"Vector store saved to: {VECTORSTORE_PATH}")

except Exception as e:
    print(f"Indexing failed: {e}")
    raise