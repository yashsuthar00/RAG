import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv('.env')

db_name = "RAG-demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

# Model names
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
gemini_model = "gemini-2.0-flash-exp"  # or "gemini-pro" for stable version

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
    print(f"\nSearching for: {query}")
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not answer the question if there is no given context.
        Do not answer the question if it is not related to the context.
        Do not give recommendations to anything other than MongoDB.
        Context:
        {context}
        Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)

    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }

    # Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    response_parser = StrOutputParser()

    # RAG chain
    rag_chain = retrieve | custom_rag_prompt | llm | response_parser
    answer = rag_chain.invoke(query)
    print(answer)

# Test queries
question = "Is this person's name is parikshit? and if not.. then what is their name."
print(f"Running query: {question}")
query_data(question)

print("=" * 80)

question = "Provide the education of the person in detial with dates and all."
print(f"Running query: {question}")
query_data(question)

print("=" * 80)

question = "Also provide the skills in detial. And also just name the projects only."
print(f"Running query: {question}")
query_data(question)