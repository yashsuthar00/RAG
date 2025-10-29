import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

load_dotenv('/app/.env')

db_name = "langchain_demo"
collection_name = "chunked_data"
index = "vector_index"
connection_string = os.getenv("CONNECTION_STRING")

# Free model names
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
llm_repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Connect to MongoDB Atlas
vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string,
    f"{db_name}.{collection_name}",
    embedding=embeddings,
    index_name=index,
)

def query_data(query):
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

    # Free LLM from Hugging Face
    llm = HuggingFaceHub(repo_id=llm_repo_id, model_kwargs={"temperature": 0})

    response_parser = StrOutputParser()

    # RAG chain
    rag_chain = retrieve | custom_rag_prompt | llm | response_parser
    answer = rag_chain.invoke(query)
    print(answer)

# Test queries
question = "When did MongoDB begin supporting multi-document transactions?"
print(f"Running query: {question}")
query_data(question)

print("=========================================================")

question = "Why is the sky blue?"
print(f"Running query: {question}")
query_data(question)