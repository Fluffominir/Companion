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