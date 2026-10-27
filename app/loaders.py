from langchain_community.document_loaders import PyPDFLoader


def loader_func(file):
    loader = PyPDFLoader(file)

    pages = loader.load()
    print('document loaded successfully')
    print(f"number of pages :{len(pages)}")
    return pages










# print(chunks[0].page_content)
# print()
# print()
# print()
# print(chunks[0].metadata)
# print(chunks[0].metadata['page_label'])

