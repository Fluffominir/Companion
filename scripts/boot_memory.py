
import glob, os
import pypdf
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pc.Index(INDEX_NAME)

def embed(text):  
    try:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

import re
from datetime import datetime

def categorize_content(text, filename):
    """Enhanced categorization with better logic"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Priority-based categorization
    if any(word in filename_lower for word in ['journal', 'diary']):
        return 'personal_journal'
    elif any(word in filename_lower for word in ['personality', 'profile', 'michael']):
        return 'personality'
    elif any(word in filename_lower for word in ['health', 'medical', 'superbill']):
        return 'health'
    elif any(word in filename_lower for word in ['company', 'employee', 'handbook', 'rocket', 'launch']):
        return 'work'
    elif any(word in filename_lower for word in ['pitch', 'serwm', 'brand']):
        return 'projects'
    elif any(word in filename_lower for word in ['attached', 'body', 'myth', 'steal', 'show', 'emyth']):
        return 'books'
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
    """Extract text using pypdf library"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting page from {pdf_path}: {e}")
                    continue
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text(text, max_length=600):
    """Improved text chunking with better sentence boundaries"""
    if len(text) <= max_length:
        return [text]
    
    # Try to split on sentence boundaries first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 2 <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]

def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

vector_count = 0
processed_files = []

print("Starting memory synchronization...")

# Process markdown files
md_files = glob.glob("docs/**/*.md", recursive=True)
print(f"Found {len(md_files)} markdown files")

for path in md_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        
        if not txt.strip():
            continue
            
        txt = clean_text(txt)
        category = categorize_content(txt, path)
        chunks = chunk_text(txt)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            embedding = embed(chunk)
            if embedding is None:
                continue
                
            vector_id = f"{path}_{i}" if len(chunks) > 1 else path.replace('/', '_')
            metadata = {
                "text": chunk,
                "source": path,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "file_type": "markdown"
            }
            
            index.upsert([(vector_id, embedding, metadata)])
            vector_count += 1
        
        processed_files.append(path)
        print(f"Processed: {path} ({category})")
        
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Process PDF files
pdf_files = glob.glob("docs/**/*.pdf", recursive=True)
print(f"Found {len(pdf_files)} PDF files")

for path in pdf_files:
    try:
        print(f"Processing PDF: {path}")
        txt = extract_pdf_text(path)
        
        if not txt.strip():
            print(f"No text extracted from {path}")
            continue
            
        txt = clean_text(txt)
        category = categorize_content(txt, path)
        chunks = chunk_text(txt, max_length=800)  # Larger chunks for PDFs
        
        processed_chunks = 0
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            embedding = embed(chunk)
            if embedding is None:
                continue
                
            vector_id = f"{path}_{i}".replace('/', '_').replace(' ', '_')
            metadata = {
                "text": chunk,
                "source": path,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "file_type": "pdf"
            }
            
            index.upsert([(vector_id, embedding, metadata)])
            vector_count += 1
            processed_chunks += 1
        
        processed_files.append(path)
        print(f"Processed: {path} ({category}) - {processed_chunks} chunks")
        
    except Exception as e:
        print(f"Error processing PDF {path}: {e}")

total_files = len(processed_files)
print(f"\nMemory sync complete!")
print(f"Successfully processed {total_files} files")
print(f"Created {vector_count} memory vectors")
print(f"Categories used: {set(categorize_content('', f) for f in processed_files)}")
