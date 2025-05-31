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

if not openai_key or "your_openai_api_key" in openai_key:
    print("❌ OPENAI_API_KEY not set properly in .env file")
if not pinecone_key or "your_pinecone_api_key" in pinecone_key:
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
    print(f"📊 OpenAI API: {'✅ Configured' if openai_key and 'your_openai_api_key' not in openai_key else '❌ Missing'}")
    print(f"📊 Pinecone API: {'✅ Configured' if pinecone_key and 'your_pinecone_api_key' not in pinecone_key else '❌ Missing'}")

# Initialize clients
try:
    openai_client = openai.OpenAI()
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
    index = pc.Index(INDEX_NAME) if pinecone_key and "your_pinecone_api_key" not in pinecone_key else None
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
        
        q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=question)\
                                       .data[0].embedding
        res = index.query(vector=q_emb, top_k=12, namespace=NAMESPACE,
                          include_metadata=True)
        # filter weak matches
        ctx = [m for m in res.matches if m.score > 0.25][:6]
        context = "\n\n".join(f"[{m.metadata['source']}] {m.metadata['text']}"
                              for m in ctx)
        messages = [
            {"role":"system","content":
             "You are Michael's assistant. Cite answers if possible."},
            {"role":"system","content":f"Context:\n{context}"},
            {"role":"user","content":question}
        ]
        reply = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2
        ).choices[0].message.content.strip()
        sources = list({m.metadata["source"].split("#")[0] + " p" +
                        m.metadata["source"].split("#")[1][1:] for m in ctx})
        return {"answer":reply, "sources":sources}
    
    except Exception as e:
        return {"answer": f"❌ Error processing question: {str(e)}", "sources": []}
