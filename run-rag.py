from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore
from app.retriever import retrieve_documents,build_context
from app.rag import get_llm


VECTORSTORE_PATH = "vector_disk"
QUERY = "What is GraphRAG?"


try:
    # 1. Load embedding model
    embeddings = get_embeddings()

    # 2. Load existing vector store
    vectorstore = load_vectorstore(
        VECTORSTORE_PATH,
        embeddings
    )

    print("Vector store loaded")

except Exception as e:
    print(f"Error loading vector store: {e}")
    raise


try:
    # 3. Retrieve relevant documents
    results = retrieve_documents(
        vectorstore,
        QUERY,
        k=3
    )

    print(f"Number of retrieved chunks: {len(results)}")

except Exception as e:
    print(f"Error during retrieval: {e}")
    raise


try:
    # 4. Build context
    context = build_context(results)
    

    # def build_context(results):
    #     context = "\n\n".join(
    #         f"[Source {i} | Page {document.metadata['page_label']}]\n"
    #         f"{document.page_content}"
    #         for i, (document, score) in enumerate(results, start=1)
    #     )

    #     return context

except Exception as e:
    print(f"Error building context: {e}")
    raise


# 5. Build prompt
prompt = f"""
Answer the question using only the provided context.

For each factual claim, cite the specific source that supports it
using [Source 1], [Source 2], etc.

Do not cite a source unless it supports the claim.

Context:
{context}

Question:
{QUERY}

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."
"""


try:
    # 6. Get LLM
    llm = get_llm()

    # 7. Generate answer
    response = llm.invoke(prompt)

except Exception as e:
    print(f"Error during LLM call: {e}")
    raise


# 8. Display answer
print("\n--- Answer ---")
print(response.content)