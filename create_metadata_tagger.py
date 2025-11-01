from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List

def create_metadata_tagger(schema: dict, llm):
    """
    Create a metadata tagger that uses the LLM to add metadata to documents based on the provided schema.
    """
    
    # Create the prompt template for metadata tagging
    template = """
    Analyze the following document content and extract metadata according to the schema.
    
    Schema: {schema}
    
    Document content:
    {content}
    
    Provide the metadata as a JSON object matching the schema. Only include the properties defined in the schema.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Create the output parser
    parser = JsonOutputParser()
    
    # Create the chain
    chain = (
        {"content": RunnablePassthrough(), "schema": lambda x: schema}
        | prompt
        | llm
        | parser
    )
    
    class MetadataTagger:
        def transform_documents(self, documents: List[Document]) -> List[Document]:
            transformed_docs = []
            for doc in documents:
                try:
                    # Get metadata from LLM
                    metadata = chain.invoke(doc.page_content)
                    
                    # Create new document with metadata
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, **metadata}
                    )
                    transformed_docs.append(new_doc)
                except Exception as e:
                    print(f"Error tagging document: {e}")
                    # If tagging fails, keep original document
                    transformed_docs.append(doc)
            
            return transformed_docs
    
    return MetadataTagger()