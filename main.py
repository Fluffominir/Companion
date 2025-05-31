import os, re
from typing import List
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import openai, pinecone
from dotenv import load_dotenv

load_dotenv()

# Check API keys
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

openai_configured = openai_key and "your_openai_api_key" not in openai_key
pinecone_configured = pinecone_key and "your_pinecone_api_key" not in pinecone_key

if not openai_configured:
    print("❌ OPENAI_API_KEY not set properly in .env file")
if not pinecone_configured:
    print("❌ PINECONE_API_KEY not set properly in .env file")

app = FastAPI(title="AI Companion", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

INDEX_NAME = "companion-memory"
NAMESPACE  = "v1"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = os.getenv("COMPANION_VOICE_MODEL", "gpt-4o-mini")

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting AI Companion...")
    print(f"📊 OpenAI API: {'✅ Configured' if openai_configured else '❌ Missing'}")
    print(f"📊 Pinecone API: {'✅ Configured' if pinecone_configured else '❌ Missing'}")

# Initialize clients
openai_client = None
index = None

try:
    if openai_configured:
        openai_client = openai.OpenAI()
        print("✅ OpenAI client initialized")
    
    if pinecone_configured:
        pc = pinecone.Pinecone(api_key=pinecone_key)
        index = pc.Index(INDEX_NAME)
        print("✅ Pinecone client initialized")
        
except Exception as e:
    print(f"❌ Error initializing APIs: {e}")
    openai_client = None
    index = None

class Answer(BaseModel):
    answer: str
    sources: List[str]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if not index:
            return {"status": "unhealthy", "error": "Pinecone not configured"}
        
        # Check if we can connect to Pinecone
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "vector_count": stats.total_vector_count,
            "index_name": INDEX_NAME,
            "openai_configured": openai_client is not None,
            "pinecone_configured": index is not None
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/ask", response_model=Answer)
async def ask(question: str = Query(..., description="Your question")):
    try:
        if not openai_client:
            return {"answer": "❌ OpenAI API not configured. Please set OPENAI_API_KEY in .env file.", "sources": []}
        
        if not index:
            return {"answer": "❌ Pinecone not configured. Please set PINECONE_API_KEY in .env file.", "sources": []}
        
        # Generate embedding
        q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
        
        # Query vector database
        res = index.query(
            vector=q_emb, 
            top_k=12, 
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        # Filter weak matches and build context
        ctx = [m for m in res.matches if m.score > 0.25][:6]
        
        if not ctx:
            return {"answer": "I don't have relevant information to answer that question. Try asking something else about your personal data.", "sources": []}
        
        context = "\n\n".join(f"[{m.metadata['source']}] {m.metadata['text']}" for m in ctx)
        
        messages = [
            {"role": "system", "content": "You are Michael's personal assistant. Use the provided context to answer questions accurately. Cite sources when possible. If you don't have relevant information, say so clearly."},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": question}
        ]
        
        # Generate response
        reply = openai_client.chat.completions.create(
            model=CHAT_MODEL, 
            messages=messages, 
            temperature=0.2
        ).choices[0].message.content.strip()
        
        # Extract sources
        sources = []
        for m in ctx:
            source = m.metadata.get("source", "")
            if "#" in source:
                source_parts = source.split("#")
                formatted_source = source_parts[0] + " p" + source_parts[1][1:]
                sources.append(formatted_source)
            else:
                sources.append(source)
        
        return {"answer": reply, "sources": list(set(sources))}
    
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return {"answer": f"❌ Error processing question: {str(e)}", "sources": []}
