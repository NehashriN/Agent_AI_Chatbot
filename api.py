import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Classic LangChain Imports
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

# 3. Initialize FastAPI
app = FastAPI(title="Gemini 3 PDF Chatbot")

# --- GLOBAL AI OBJECT ---
qa_chain = None

# --- THE REDIRECT ROUTE ---
# This ensures you don't see a blank "Not Found" page at the root link
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# --- AI INITIALIZATION ---
def initialize_ai():
    global qa_chain
    
    google_key = os.getenv("GOOGLE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not all([google_key, pinecone_key, index_name]):
        print("❌ ERROR: Missing API keys in .env file.")
        return

    try:
        # 2026 Standard Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_key
        )
        
        # 2026 Standard LLM (Gemini 3 Flash)
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=google_key,
            temperature=0
        )

        vectorstore = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings,
            pinecone_api_key=pinecone_key
        )

        # Build the Classic Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("✅ SUCCESS: AI Brain loaded and ready!")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

# Run the setup on startup
initialize_ai()

# --- API ENDPOINTS ---
class ChatRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_pdf(request: ChatRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="AI Brain not loaded.")
    
    try:
        # Using the Classic .run() method
        response = qa_chain.run(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


