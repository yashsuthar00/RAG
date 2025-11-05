import os
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv('.env')

db_name = "RAG-demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

# Free model names
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Embeddings
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Connect to MongoDB Atlas
print("Connecting to MongoDB Atlas...")
vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string,
    f"{db_name}.{collection_name}",
    embedding=embeddings,
    index_name=index,
)

def query_data(query):
    """Simple retrieval without LLM - just returns relevant documents"""
    print(f"\nSearching for: {query}")
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    
    results = retriever.invoke(query)
    
    print(f"\nFound {len(results)} relevant documents:")
    print("=" * 80)
    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content[:500]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 80)
    
    return results

# Test queries
question = "Provide skills of the person."
print(f"Running query: {question}")
query_data(question)
