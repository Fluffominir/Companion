import glob, os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)

def embed(text):  
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

import re
from datetime import datetime

def categorize_content(text, filename):
    """Categorize content based on keywords and structure"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    if any(word in text_lower for word in ['goal', 'objective', 'want to', 'plan to', 'achieve']):
        return 'goals'
    elif any(word in text_lower for word in ['meeting', 'call', 'appointment', 'schedule']):
        return 'meetings'
    elif any(word in text_lower for word in ['project', 'task', 'todo', 'work on']):
        return 'projects'
    elif any(word in text_lower for word in ['idea', 'thought', 'concept', 'brainstorm']):
        return 'ideas'
    elif any(word in filename_lower for word in ['personal', 'diary', 'journal']):
        return 'personal'
    else:
        return 'general'

def chunk_text(text, max_length=500):
    """Split long text into smaller chunks"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

vector_count = 0
for path in glob.glob("docs/**/*.md", recursive=True):
    with open(path) as f:
        txt = f.read()
    
    category = categorize_content(txt, path)
    chunks = chunk_text(txt)
    
    for i, chunk in enumerate(chunks):
        vector_id = f"{path}_{i}" if len(chunks) > 1 else path
        metadata = {
            "text": chunk,
            "source": path,
            "category": category,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat()
        }
        index.upsert([(vector_id, embed(chunk), metadata)])
        vector_count += 1

print(f"Synced {vector_count} vectors from {len(glob.glob('docs/**/*.md', recursive=True))} documents")
