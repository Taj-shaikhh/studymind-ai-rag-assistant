from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq

load_dotenv()

data = TextLoader("document loaders/notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [('system',"You are an AI that summarizes the text"),
     ('human',"{data}")]
)

# .from_messages() is a class method that creates a ChatPromptTemplate from a list of messages, such as system, human, and AI messages.

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt = template.format_messages(data = docs[0].page_content)

response = model.invoke(prompt)
print(response.content)