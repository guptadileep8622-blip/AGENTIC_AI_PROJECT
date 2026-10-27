def retrieve_documents(vectorstore, query, k=3):
    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    return results


def build_context(results):
    context = "\n\n".join(
        f"[Source {i} | Page {document.metadata['page_label']}]\n"
        f"{document.page_content}"
        for i, (document, score) in enumerate(results, start=1)
    )

    return context