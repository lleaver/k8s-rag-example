""" 
Simple pipeline for testing RAG pipeline with NIM microservices on K8s
Uses LLM and Embedder NIMs, NVIDIA langchain connectors and FAISS vector store
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

data_file = "example.txt" #add path to your example data!

loader = TextLoader(data_file)
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
docs = loader.load_and_split(text_splitter=splitter)

## Define pipeline components: LLM, Embedder
# add base_url to these to connect to local
llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",
    base_url="http://<worker-node>:31001", #add address of your worker node!
)

embedder = NVIDIAEmbeddings(
  model="nvidia/nv-embedqa-e5-v5",
  base_url="http://<worker-node>:31002", #add address of your worker node!
  truncate="NONE",
)

## create vector store and add docs
db = FAISS.from_documents(docs, embedder)

## Define compression retriever for retrieval + reranking
retriever = db.as_retriever()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

## define chain
chain = llm | StrOutputParser()

def ask_question(question, context):
    """returns a response from the llm"""
    full_question = ("context: " + str(context) + "question :" + str(question))
    ans = llm.invoke(full_question)
    return ans.content

q = "What is the main topic of the provided context?"
context = format_docs(retriever.invoke(q))
print(ask_question(q, context))