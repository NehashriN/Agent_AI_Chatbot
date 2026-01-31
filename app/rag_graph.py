from typing import List, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from app.retriever import get_retriever
from app.generator import get_generation_chain

# This is the 'State' - the data that moves through our graph
class GraphState(TypedDict):
    question: str
    generation: str
    context: List[str]
    confidence: float

# A simple structure to hold our 'grading' results
class GradeResult(BaseModel):
    is_grounded: bool = Field(description="Is the answer strictly based on context?")
    score: float = Field(description="Confidence score (0-100)")

# NODE 1: Retrieval
def retrieve_node(state: GraphState):
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"context": [d.page_content for d in docs]}

# NODE 2: Generation
def generate_node(state: GraphState):
    chain = get_generation_chain()
    response = chain.invoke({
        "context": "\n\n".join(state["context"]),
        "question": state["question"]
    })
    return {"generation": response.content}

# NODE 3: Confidence Grading
def grade_node(state: GraphState):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    # Force the AI to return a specific 'GradeResult' format
    scorer = llm.with_structured_output(GradeResult)
    
    eval_query = f"Context: {state['context']}\nAnswer: {state['generation']}"
    result = scorer.invoke(f"Is this answer supported by the context? {eval_query}")
    
    # If the AI says it's not grounded, we set confidence to 0
    return {"confidence": result.score if result.is_grounded else 0.0}

# --- BUILD THE GRAPH ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade", grade_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")
workflow.add_edge("grade", END)

# This 'app' is what we will call from our API
rag_app = workflow.compile()