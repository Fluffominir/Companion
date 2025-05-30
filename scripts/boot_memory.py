import glob, os
import PyPDF2
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
    
    # File-based categorization first
    if any(word in filename_lower for word in ['journal', 'diary']):
        return 'personal_journal'
    elif any(word in filename_lower for word in ['personality', 'profile']):
        return 'personality'
    elif any(word in filename_lower for word in ['health', 'medical', 'superbill']):
        return 'health'
    elif any(word in filename_lower for word in ['company', 'employee', 'handbook', 'rocket']):
        return 'work'
    elif any(word in filename_lower for word in ['pitch', 'serwm', 'brand']):
        return 'projects'
    elif any(word in filename_lower for word in ['attached', 'body', 'myth', 'steal', 'show']):
        return 'books'
    
    # Content-based categorization
    elif any(word in text_lower for word in ['goal', 'objective', 'want to', 'plan to', 'achieve']):
        return 'goals'
    elif any(word in text_lower for word in ['meeting', 'call', 'appointment', 'schedule']):
        return 'meetings'
    elif any(word in text_lower for word in ['project', 'task', 'todo', 'work on']):
        return 'projects'
    elif any(word in text_lower for word in ['idea', 'thought', 'concept', 'brainstorm']):
        return 'ideas'
    else:
        return 'general'

def extract_pdf_text(pdf_path):
    """Extract text from PDF files"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

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

# Process markdown files
md_files = glob.glob("docs/**/*.md", recursive=True)
for path in md_files:
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
            "timestamp": datetime.now().isoformat(),
            "file_type": "markdown"
        }
        index.upsert([(vector_id, embed(chunk), metadata)])
        vector_count += 1

# Process PDF files
pdf_files = glob.glob("docs/**/*.pdf", recursive=True)
for path in pdf_files:
    print(f"Processing PDF: {path}")
    txt = extract_pdf_text(path)
    
    if txt.strip():  # Only process if we extracted text
        category = categorize_content(txt, path)
        chunks = chunk_text(txt, max_length=800)  # Larger chunks for PDFs
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                vector_id = f"{path}_{i}" if len(chunks) > 1 else path
                metadata = {
                    "text": chunk,
                    "source": path,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat(),
                    "file_type": "pdf"
                }
                index.upsert([(vector_id, embed(chunk), metadata)])
                vector_count += 1

total_files = len(md_files) + len(pdf_files)
print(f"Synced {vector_count} vectors from {total_files} documents ({len(md_files)} MD, {len(pdf_files)} PDF)")
