from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# PyPDFLoader reads text from a PDF file and converts it into LangChain Document objects so you can process the content using an AI model.

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()
# load() is a method that reads the file and returns its content as a list of LangChain Document objects.

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)