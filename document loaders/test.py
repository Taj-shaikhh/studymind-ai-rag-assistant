from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("document loaders/notes.txt")
docs = data.load()

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 1000,
    chunk_overlap = 1
)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)