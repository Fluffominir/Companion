import os, re
from typing import List
from fastapi import FastAPI, Query
from pydantic import BaseModel
import openai, pinecone, dotenv

dotenv.load_dotenv()
app = FastAPI()

INDEX_NAME = "companion-memory"
NAMESPACE  = "v1"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = os.getenv("COMPANION_VOICE_MODEL", "gpt-4o-mini")

openai_client = openai.OpenAI()
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

class Answer(BaseModel):
    answer: str
    sources: List[str]

@app.get("/ask", response_model=Answer)
async def ask(question: str = Query(..., description="Your question")):
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
