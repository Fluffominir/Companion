"""ingest_simple.py — Simple PDF extraction and embedding for Replit
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
            page_text = page.extract_text() or ""
            text += page_text

        # Clean up text encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
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

                # Clean chunk text
                chunk = chunk.strip()
                if len(chunk) > 8000:  # OpenAI embedding limit
                    chunk = chunk[:8000]

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
                    print(f"  ❌ Error embedding chunk {i}: {e}")
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
"""
ingest_simple.py
Simple document ingestion for personal documents
"""
import os, pathlib, json, uuid
from typing import List
import openai, pinecone
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Constants
EMBED_MODEL = "text-embedding-3-small"
INDEX_NAME = "companion-memory"
NAMESPACE = "v1"


def ingest_markdown(md_file: pathlib.Path) -> List[dict]:
    """Extract text from Markdown file and create chunks"""
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = [{
            'id': f"{md_file.stem}_full",
            'text': text,
            'source': md_file.name,
            'metadata': {
                'source': md_file.name,
                'text': text[:300],
                'document': md_file.stem,
            }
        }]
        return chunks
    except Exception as e:
        print(f"Error reading markdown {md_file.name}: {e}")
        return []


def ingest_pdf(pdf_path: pathlib.Path) -> List[dict]:
    """Extract text from PDF and create chunks"""
    try:
        reader = PdfReader(str(pdf_path))
        chunks = []

        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    # Simple chunking by paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    for j, para in enumerate(paragraphs):
                        if len(para) > 100:  # Skip very short paragraphs
                            chunk_id = f"{pdf_path.stem}_p{i}_c{j}"
                            chunks.append({
                                'id': chunk_id,
                                'text': para,
                                'source': f"{pdf_path.name}#p{i}",
                                'metadata': {
                                    'source': f"{pdf_path.name}#p{i}",
                                    'text': para[:300],  # First 300 chars for metadata
                                    'document': pdf_path.stem,
                                    'page': i,
                                    'chunk': j
                                }
                            })
            except Exception as e:
                print(f"Error processing page {i} of {pdf_path.name}: {e}")
                continue
        return chunks
    except Exception as e:
        print(f"Error reading PDF {pdf_path.name}: {e}")
        return []


def main():
    """Main ingestion process"""
    load_dotenv()

    # Check environment variables
    if not os.environ.get("OPENAI_API_KEY") or "your_openai_api_key" in os.environ.get("OPENAI_API_KEY", ""):
        print("❌ Please set your OPENAI_API_KEY in .env file")
        return

    if not os.environ.get("PINECONE_API_KEY") or "your_pinecone_api_key" in os.environ.get("PINECONE_API_KEY", ""):
        print("❌ Please set your PINECONE_API_KEY in .env file")
        return

    # Initialize OpenAI and Pinecone
    try:
        openai_client = openai.OpenAI()
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    except Exception as e:
        print(f"❌ Error initializing APIs: {e}")
        return

    # Create or get index
    try:
        index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to existing index: {INDEX_NAME}")
    except:
        try:
            print(f"📁 Creating new index: {INDEX_NAME}")
            pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine")
            index = pc.Index(INDEX_NAME)
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return

    # Process all documents
    all_chunks = []
    total_chunks = 0

    # Process PDFs from attached_assets
    assets_dir = pathlib.Path("attached_assets")
    if assets_dir.exists():
        pdf_files = list(assets_dir.glob("*.pdf"))
        print(f"📄 Found {len(pdf_files)} PDF files in attached_assets/")
        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file.name}")
            chunks = ingest_pdf(pdf_file)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            print(f"    ✅ Extracted {len(chunks)} chunks")

    # Process docs directory
    docs_dir = pathlib.Path("docs")
    if docs_dir.exists():
        md_files = list(docs_dir.glob("*.md"))
        print(f"📝 Found {len(md_files)} Markdown files in docs/")
        for file in md_files:
            print(f"  Processing: {file.name}")
            chunks = ingest_markdown(file)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            print(f"    ✅ Extracted {len(chunks)} chunks")

    if total_chunks == 0:
        print("❌ No content found to ingest. Add PDF files to attached_assets/ or markdown files to docs/")
        return

    # Embed and upsert to Pinecone
    print(f"🧠 Embedding and uploading {total_chunks} chunks...")
    successful_uploads = 0

    for i, chunk in enumerate(all_chunks):
        try:
            # Create embedding
            embedding = openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk['text']
            ).data[0].embedding

            # Upsert to Pinecone
            index.upsert(vectors=[{
                'id': chunk['id'],
                'values': embedding,
                'metadata': chunk['metadata']
            }], namespace=NAMESPACE)

            successful_uploads += 1
            if (i + 1) % 10 == 0:
                print(f"  📤 Uploaded {i + 1}/{total_chunks} chunks...")

        except Exception as e:
            print(f"❌ Error processing chunk {chunk['id']}: {e}")

    print(f"✅ Ingestion complete! Successfully uploaded {successful_uploads}/{total_chunks} chunks")

if __name__ == "__main__":
    main()