from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def get_generation_chain():
    # Initialize Gemini 1.5 Flash (free/cheap and fast)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # The 'Grounding' instruction: tell the AI it CANNOT guess
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the context below to answer.
    If the answer isn't in the context, say: "I cannot find this in the eBook."
    
    CONTEXT:
    {context}

    QUESTION: 
    {question}
    """)
    
    # Combine the prompt and the LLM into a single 'chain'
    return prompt | llm