from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    print('embeddings generated')
    return embeddings









