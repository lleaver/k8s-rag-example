""" 
Simple pipeline for testing RAG pipeline with NIM microservices on K8s
Uses LLM and Embedder NIMs, NVIDIA langchain connectors and FAISS vector store
"""

import getpass

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

#get api key
password = getpass.getpass("Enter your NVAPI Personal Key: ")
nvapi_key = password

## Load data
data_file = "path to .txt file here"

loader = TextLoader(data_file)
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
docs = loader.load_and_split(text_splitter=splitter)

## Define pipeline components: LLM, Embedder
# add base_url to these to connect to local
llm = ChatNVIDIA(
    model="meta/llama2-70b", 
    nvidia_api_key=nvapi_key
)

embedder = NVIDIAEmbeddings(
  model="nvidia/nv-embedqa-e5-v5", 
  api_key=nvapi_key, 
  truncate="NONE", 
)

## create vector store and add docs
db = FAISS.from_documents(docs, embedder)

## Define compression retriever for retrieval + reranking
retriever = db.as_retriever()

## define chain
chain = llm | StrOutputParser()

def ask_question(question):
    """returns a response from the llm"""
    ans = chain.invoke(question)
    return ans

q = "What is the main topic of the provided context?"
print(ask_question(q))