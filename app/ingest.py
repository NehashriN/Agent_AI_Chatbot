import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def run_ingestion():
    # Connect to Pinecone using your secret API key
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "agentic-ai-index"

    # Create a 'filing cabinet' (Index) in Pinecone if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Gemini embeddings use 768 dimensions
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    # 1. Load: Get the PDF from the web
    loader = PyPDFLoader("https://konverge.ai/pdf/Ebook-Agentic-AI.pdf")
    data = loader.load()

    # 2. Chunk: Split the big PDF into smaller paragraphs (700 characters each)
    # This helps the AI find specific answers faster
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    chunks = text_splitter.split_documents(data)

    # 3. Embed & Store: Turn text into numbers (vectors) and save to Pinecone
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    print(f"Saving {len(chunks)} chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    print("Success! The eBook is now in your database.")