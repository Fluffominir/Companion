import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set")

INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Failed to connect to Pinecone index: {e}")
    raise Exception(f"Failed to connect to Pinecone index: {e}")

app = FastAPI(title="Michael's AI Companion")

# Simple conversation memory
conversation_memory = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

def embed_text(text: str) -> List[float]:
    """Create embeddings for text"""
    try:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Error creating embedding")

def retrieve_context(query: str, top_k: int = 8) -> List[Dict]:
    """Retrieve relevant context from vector database"""
    try:
        logger.info(f"Searching for: '{query}'")

        # Get embeddings and search
        query_embedding = embed_text(query)
        results = index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Get more to filter
            include_metadata=True
        ).matches

        # Filter and deduplicate results
        filtered_results = []
        seen_content = set()

        for result in results:
            if result.score < 0.3:  # Skip low-relevance results
                continue

            text_content = result.metadata.get('text', '')
            content_preview = text_content[:100]

            if content_preview in seen_content:
                continue

            seen_content.add(content_preview)
            filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        logger.info(f"Found {len(filtered_results)} relevant results")
        return filtered_results

    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        return []

def format_context(results: List[Dict]) -> str:
    """Format retrieved context for the AI"""
    if not results:
        return ""

    context_parts = ["RELEVANT INFORMATION FROM MICHAEL'S PERSONAL DATA:"]

    for i, result in enumerate(results, 1):
        metadata = result.metadata
        source = Path(metadata.get('source', 'unknown')).name
        category = metadata.get('category', 'general')
        text = metadata.get('text', '')
        score = result.score

        context_parts.append(f"\n{i}. SOURCE: {source} (Category: {category}, Relevance: {score:.3f})")
        context_parts.append(f"   CONTENT: {text}")

    context_parts.append("\nIMPORTANT: Only use information explicitly stated above. Do not infer or guess details not present in this context.")

    return "\n".join(context_parts)

@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint with RAG"""
    try:
        logger.info(f"Chat request from {req.session_id}: {req.message[:100]}...")

        # Retrieve relevant context
        context_results = retrieve_context(req.message)
        context = format_context(context_results)

        # Manage conversation memory
        if req.session_id not in conversation_memory:
            conversation_memory[req.session_id] = []

        # Build messages for AI
        system_prompt = """You are Michael's personal AI companion with access to his comprehensive personal database.

ACCURACY RULES - CRITICAL:
🚫 NEVER guess, assume, or fabricate personal information
🚫 NEVER state personal details unless explicitly found in the provided context
🚫 If you cannot find specific information, say: "I don't see that information in your documented data. Could you help me understand what you're looking for?"

✅ ONLY state facts that are explicitly written in the context
✅ Always cite the source when sharing personal information
✅ When uncertain, ask clarifying questions
✅ Be helpful and conversational while maintaining accuracy

You have access to Michael's journals, personality assessments, work information, goals, and personal data."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add context if available
        if context:
            messages.append({"role": "system", "content": context})

        # Add recent conversation history (last 6 messages)
        recent_conversation = conversation_memory[req.session_id][-6:]
        messages.extend(recent_conversation)

        # Add current message
        messages.append({"role": "user", "content": req.message})

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        response_content = response.choices[0].message.content

        # Update conversation memory
        conversation_memory[req.session_id].append({"role": "user", "content": req.message})
        conversation_memory[req.session_id].append({"role": "assistant", "content": response_content})

        # Keep memory manageable
        if len(conversation_memory[req.session_id]) > 20:
            conversation_memory[req.session_id] = conversation_memory[req.session_id][-16:]

        return {"response": response_content}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/status")
async def api_status():
    """API health check with system status"""
    try:
        # Check Pinecone connection
        stats = index.describe_index_stats()

        # Quick search test
        test_results = index.query(
            vector=[0.0] * 1536,
            top_k=10,
            include_metadata=True
        )

        categories = {}
        for match in test_results.matches:
            cat = match.metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "status": "healthy",
            "message": "AI Companion is running",
            "total_memory_vectors": stats.total_vector_count,
            "categories": categories,
            "features": [
                "Advanced RAG retrieval",
                "Personal data processing",
                "Conversation memory",
                "Accuracy-focused responses"
            ]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": "API running with limited functionality",
            "error": str(e)
        }

@app.get("/api/debug/search")
async def debug_search(q: str = "family"):
    """Debug endpoint to test search functionality"""
    try:
        results = retrieve_context(q, top_k=5)
        context = format_context(results)

        return {
            "query": q,
            "results_found": len(results),
            "context": context,
            "result_details": [
                {
                    "source": Path(r.metadata.get('source', '')).name,
                    "category": r.metadata.get('category', ''),
                    "score": r.score,
                    "text_preview": r.metadata.get('text', '')[:200]
                } for r in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Simple health check for monitoring"""
    return {"status": "ok"}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)