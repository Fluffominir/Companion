
"""
ingest_simple.py — Simple PDF extraction and embedding for Replit
Based on the uploaded companion setup but adapted for your existing structure
"""

import os, json, re, uuid, datetime as dt, pathlib
from pypdf import PdfReader
import openai, dotenv
import pinecone

# Load environment
dotenv.load_dotenv()
client = openai.OpenAI()
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("companion-memory")

# Use your existing directory structure
SOURCE_DIRS = ["docs/", "attached_assets/"]
NAMESPACE = "v1"

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Split text into chunks"""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:  # Skip very short paragraphs
            continue
            
        # If paragraph is too long, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'[.!?]+', para)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += sentence + ". "
                    
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            chunks.append(para)
    
    return chunks

def main():
    print("🚀 Starting simple ingestion...")
    
    total_chunks = 0
    processed_files = 0
    
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            continue
            
        for pdf_file in pathlib.Path(source_dir).glob("*.pdf"):
            print(f"📄 Processing: {pdf_file.name}")
            
            # Extract text
            text = extract_pdf_text(pdf_file)
            if not text.strip():
                print(f"  ⚠️  No text found in {pdf_file.name}")
                continue
                
            # Chunk text
            chunks = chunk_text(text)
            print(f"  📝 Created {len(chunks)} chunks")
            
            # Process each chunk
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                    
                # Generate embedding
                try:
                    embedding = client.embeddings.create(
                        model="text-embedding-3-small", 
                        input=chunk
                    ).data[0].embedding
                    
                    # Create vector
                    chunk_id = f"{pdf_file.stem}_{i}_{uuid.uuid4().hex[:6]}"
                    vectors_to_upsert.append({
                        "id": chunk_id,
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "source": pdf_file.name,
                            "category": "personal_database",
                            "file_type": "pdf"
                        }
                    })
                    
                except Exception as e:
                    print(f"  ❌ Error embedding chunk: {e}")
                    continue
            
            # Upsert to Pinecone
            if vectors_to_upsert:
                try:
                    index.upsert(vectors_to_upsert, namespace=NAMESPACE)
                    total_chunks += len(vectors_to_upsert)
                    processed_files += 1
                    print(f"  ✅ Uploaded {len(vectors_to_upsert)} vectors")
                except Exception as e:
                    print(f"  ❌ Error uploading to Pinecone: {e}")
    
    print(f"\n🎉 Ingestion complete!")
    print(f"Files processed: {processed_files}")
    print(f"Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()
