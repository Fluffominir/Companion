import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set")

INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
except Exception as e:
    raise Exception(f"Failed to create Pinecone index: {e}")


index = pc.Index(INDEX_NAME)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def embed(text: str) -> List[float]:
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def fetch_memories(q: str, k: int = 5) -> List[Dict]:
    """Fetch relevant memories with enhanced filtering"""
    results = index.query(
        vector=embed(q), 
        top_k=k, 
        include_metadata=True,
        filter={}  # You can add filters here based on category, date, etc.
    ).matches
    return results

def format_memory_context(memories: List[Dict]) -> str:
    """Format memories into a coherent context"""
    if not memories:
        return ""

    context_parts = []
    categories = {}

    # Group memories by category
    for memory in memories:
        metadata = memory.metadata
        category = metadata.get('category', 'general')
        if category not in categories:
            categories[category] = []
        categories[category].append(metadata.get('text', ''))

    # Format by category
    for category, texts in categories.items():
        if category != 'general':
            context_parts.append(f"\n{category.upper()} NOTES:")
            for text in texts:
                snippet = text[:200] + "..." if len(text) > 200 else text
                context_parts.append(f"- {snippet}")

    return "\n".join(context_parts)

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        memories = fetch_memories(req.message)
        memory_context = format_memory_context(memories)

        system_prompt = """You are Michael's helpful personal AI companion. You have access to Michael's notes, thoughts, and information.

Key traits:
- Be conversational and supportive
- Remember context from previous interactions
- Help with goals, projects, and daily tasks
- Provide insights based on stored memories
- Ask clarifying questions when helpful

Use the memories below to provide relevant, personalized assistance."""

        msgs = [{"role": "system", "content": system_prompt}]

        if memory_context:
            msgs.append({
                "role": "system", 
                "content": f"RELEVANT MEMORIES:\n{memory_context}"
            })

        msgs.append({"role": "user", "content": req.message})

        resp = client.chat.completions.create(
            model="gpt-4o", 
            messages=msgs,
            temperature=0.7,
            max_tokens=500
        )
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API is running"}