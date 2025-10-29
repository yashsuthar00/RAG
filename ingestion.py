import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from create_metadata_tagger import create_metadata_tagger
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from pymongo import MongoClient

load_dotenv('/app/.env')

# Model names
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_repo_id = "HuggingFaceH4/zephyr-7b-beta"

# MongoDB setup
client = MongoClient(os.getenv("CONNECTION_STRING"))
collection = client["RAG-demo"]["chunked_data"]

print("Deleting the collection before adding new data")
collection.delete_many({})

# Load and clean PDF
loader = PyPDFLoader("./cv/cv.pdf")
pages = loader.load()
cleaned_pages = [p for p in pages if len(p.page_content.split(" ")) > 20]

print("Splitting the documents into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# Metadata tagging schema
schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

print("Creating metadata for the documents")
llm = HuggingFaceHub(repo_id=llm_repo_id, model_kwargs={"temperature": 0})
document_transformer = create_metadata_tagger(schema, llm)
docs = document_transformer.transform_documents(cleaned_pages)
split_docs = text_splitter.split_documents(docs)

# Embeddings
print("Generating embeddings using Hugging Face model")
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Store vectors
print("Storing the vectors in MongoDB Atlas")
vector_store = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embedding, collection=collection
)

document_count = collection.count_documents({})
print(f"✅ Successfully stored {document_count} documents in MongoDB Atlas")
client.close()