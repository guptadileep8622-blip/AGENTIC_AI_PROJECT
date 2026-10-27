from langchain_community.vectorstores import FAISS



def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    print('vector is stored now')
    return vectorstore

def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)

def load_vectorstore(path, embeddings):
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore   










