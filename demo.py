from dotenv import load_dotenv
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv('.env')

db_name = "RAG-demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Connect to vector store
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=connection_string,
    namespace=f"{db_name}.{collection_name}",
    embedding=HuggingFaceEmbeddings(model_name=embedding_model_name),
    index_name=index,
)

def query_data(query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    results = retriever.invoke(query)
    return results
# Example query
print(query_data("When did MongoDB begin supporting multi-document transactions?"))