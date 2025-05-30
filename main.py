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

def fetch_memories(q: str, k: int = 3) -> List[Dict]:
    return index.query(vector=embed(q), top_k=k, include_metadata=True).matches

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        memories = fetch_memories(req.message)
        msgs = [
            {
                "role": "system",
                "content": "You are Michael's helpful personal assistant.",
            }
        ]
        for m in memories:
            snippet = m.metadata.get("text", "")[:150]
            msgs.append({"role": "system", "content": f"(Memory): {snippet}"})
        msgs.append({"role": "user", "content": req.message})
        resp = client.chat.completions.create(model="gpt-4o", messages=msgs)
        return {"response": resp.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")