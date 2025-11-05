import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv('.env')

# Connect to MongoDB
client = MongoClient(os.getenv("CONNECTION_STRING"))
db = client["RAG-demo"]
collection = db["chunked_data"]

# Check documents
count = collection.count_documents({})
print(f"Total documents in collection: {count}")

if count > 0:
    print("\nSample document:")
    sample = collection.find_one()
    print(sample)
    
    print("\n" + "="*80)
    print("IMPORTANT: You need to create a Vector Search Index in MongoDB Atlas!")
    print("="*80)
    print("\nSteps to create the index:")
    print("1. Go to your MongoDB Atlas dashboard")
    print("2. Navigate to your cluster")
    print("3. Click on 'Search' tab")
    print("4. Click 'Create Search Index'")
    print("5. Choose 'JSON Editor'")
    print("6. Use this configuration:\n")
    
    index_definition = """{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}"""
    
    print(index_definition)
    print("\n7. Set the index name to: vector_index")
    print("8. Set the database to: RAG-demo")
    print("9. Set the collection to: chunked_data")
    print("10. Click 'Create Search Index'")
    print("\nAfter creating the index, wait a few minutes for it to build, then run retriver_simple.py again.")
    
else:
    print("\nNo documents found. Run ingestion.py first!")

client.close()
