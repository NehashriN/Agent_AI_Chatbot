🤖 Gemini 3 PDF Chatbot (RAG + LangGraph)
A professional-grade Retrieval-Augmented Generation (RAG) system utilizing Gemini 3 Flash, Pinecone, and LangGraph for reliable, grounded document intelligence.

🚀 1. Setup Instructions
Prerequisites
Python Version: 3.10 or higher (recommended 3.12 for 2026 performance).

API Keys: Google AI Studio (Gemini 3), Pinecone, and LangChain (optional for tracing).

Installation
Clone and Enter:

Bash
git clone https://github.com/your-username/gemini-pdf-chatbot.git
cd gemini-pdf-chatbot
Install Dependencies:

Bash
pip install -r requirements.txt
Includes: fastapi, langchain-classic, langchain-google-genai, langchain-pinecone, python-dotenv.

Environment Setup (.env): Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=AIza...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=your-index-name
🏗️ 2. Architecture Overview
This project implements a Self-Reflective RAG architecture. Unlike a linear pipeline, we use LangGraph to create a stateful, cyclic workflow that verifies the quality of retrieved information before the final answer is generated.

The Data Flow:

Ingestion: PDF text is parsed, chunked, and stored as vectors in Pinecone.

Retrieval: User queries are embedded to fetch the top-k relevant document chunks.

Refinement (LangGraph): The system "grades" the retrieved chunks for relevance. If irrelevant, it re-formulates the query.

Generation: Only "vetted" chunks are passed to Gemini 3 Flash to produce the grounded response.

📐 3. Design Decisions
Why Gemini 3 Flash?
We chose the gemini-3-flash-preview (2026) for its high-speed inference and massive context window. It provides the best cost-to-intelligence ratio for real-time PDF chat.

Chunking Strategy: Recursive Character (Size: 1000, Overlap: 200)
Size (1000 tokens): Large enough to capture complete paragraphs and technical context, but small enough to fit 5-10 chunks into the prompt without noise.

Overlap (200 tokens): Essential for maintaining semantic continuity. It ensures that if an answer is split between two chunks, the "bridge" of information is preserved.

Embedding Model: text-embedding-004
Selected for its high dimensionality and native compatibility with the Google ecosystem, ensuring that "Vector Similarity" accurately reflects "Semantic Meaning."

Why LangGraph?
Standard "Chains" are rigid. LangGraph allows us to add Conditional Edges. If the retriever finds "noise," the graph can loop back to try a different search strategy instead of giving the user a hallucinated or incorrect answer.

🛡️ 4. How Grounding is Enforced (Critical)
Grounding is the mechanism that prevents the AI from "making things up." We enforce this through three layers:

The System Constraint: The Gemini 3 system prompt is explicitly hard-coded to say: "Use ONLY the provided context to answer. If the answer is not in the context, say you do not know."

Context Injection: The user's question is wrapped in the retrieved document chunks. The LLM sees the source material as the primary truth, superseding its own internal training data.

Factual Verification (LangGraph Grade Node): Before the response is sent to you, a "hidden" validation step checks if the generated answer is supported by the source IDs retrieved from Pinecone. This eliminates "hallucination by omission."

Final Step for You
Create the file README.md in VS Code.

Paste the code above.

Run pip freeze > requirements.txt in your terminal so that the "Install dependencies" step works for anyone who clones your repo!