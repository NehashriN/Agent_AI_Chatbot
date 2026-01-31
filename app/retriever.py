import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_retriever():
    # Use the same 'math' (embeddings) we used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Connect to our existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name="agentic-ai-index", 
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # Return a tool that grabs the top 3 most relevant snippets
    return vectorstore.as_retriever(search_kwargs={"k": 3})