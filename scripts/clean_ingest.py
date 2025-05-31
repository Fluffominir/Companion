
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
import markdown
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys in environment")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize or create index
try:
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
    index = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 500
        self.overlap = 50
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
                
        return chunks
    
    def categorize_content(self, text: str, filename: str) -> str:
        """Categorize content based on filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # Priority categories for personal data
        if 'personality' in filename_lower:
            return 'personality'
        elif 'profile' in filename_lower:
            return 'profile'
        elif 'journal' in filename_lower or any(year in filename_lower for year in ['2022', '2024', '2025']):
            return 'journal'
        elif 'family' in text_lower or any(word in text_lower for word in ['mother', 'father', 'parent', 'mom', 'dad']):
            return 'family'
        elif 'goal' in filename_lower or 'goal' in text_lower:
            return 'goals'
        elif 'rocket' in filename_lower or 'work' in filename_lower:
            return 'work'
        elif 'health' in filename_lower:
            return 'health'
        else:
            return 'general'
    
    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            return ""
    
    def extract_markdown_text(self, file_path: str) -> str:
        """Extract text from markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Convert markdown to plain text (remove formatting)
            text = markdown.markdown(content)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            return text
        except Exception as e:
            logger.error(f"Error extracting markdown {file_path}: {e}")
            return ""

def embed_text(text: str) -> List[float]:
    """Create embeddings for text"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def clear_index():
    """Clear all vectors from the index"""
    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            logger.info(f"Clearing {stats.total_vector_count} existing vectors...")
            index.delete(delete_all=True)
            logger.info("Index cleared successfully")
    except Exception as e:
        logger.warning(f"Error clearing index: {e}")

def ingest_documents():
    """Main ingestion process"""
    processor = DocumentProcessor()
    vector_count = 0
    processed_files = []
    
    print("🧠 Starting clean document ingestion...")
    
    # Clear existing data
    clear_index()
    
    # Define source directories and file types
    source_dirs = ["docs/", "attached_assets/"]
    supported_files = {".pdf", ".md", ".txt"}
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        for file_path in Path(source_dir).rglob("*"):
            if file_path.suffix.lower() not in supported_files:
                continue
                
            logger.info(f"Processing: {file_path}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == ".pdf":
                text = processor.extract_pdf_text(str(file_path))
            elif file_path.suffix.lower() == ".md":
                text = processor.extract_markdown_text(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                continue
                
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                continue
            
            # Clean and process text
            text = processor.clean_text(text)
            category = processor.categorize_content(text, file_path.name)
            chunks = processor.chunk_text(text)
            
            # Create vectors for each chunk
            file_vectors = 0
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                embedding = embed_text(chunk)
                if embedding is None:
                    continue
                
                vector_id = f"{file_path.stem}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                metadata = {
                    "text": chunk,
                    "source": str(file_path),
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat(),
                    "file_type": file_path.suffix.lower()
                }
                
                try:
                    index.upsert([(vector_id, embedding, metadata)])
                    file_vectors += 1
                    vector_count += 1
                except Exception as e:
                    logger.error(f"Error upserting vector: {e}")
            
            if file_vectors > 0:
                processed_files.append({
                    "file": str(file_path),
                    "category": category,
                    "chunks": file_vectors
                })
                logger.info(f"✅ Processed {file_path.name}: {file_vectors} chunks in category '{category}'")
    
    # Summary
    print(f"\n📊 Ingestion Complete!")
    print(f"Files processed: {len(processed_files)}")
    print(f"Total vectors created: {vector_count}")
    print("\nFiles by category:")
    
    categories = {}
    for file_info in processed_files:
        cat = file_info["category"]
        if cat not in categories:
            categories[cat] = {"files": 0, "chunks": 0}
        categories[cat]["files"] += 1
        categories[cat]["chunks"] += file_info["chunks"]
    
    for category, stats in categories.items():
        print(f"  {category}: {stats['files']} files, {stats['chunks']} chunks")
    
    return processed_files, vector_count

def test_retrieval():
    """Test the retrieval system with sample queries"""
    test_queries = [
        "family members",
        "mother father parents",
        "personality traits",
        "goals objectives",
        "work rocket launch studio"
    ]
    
    print(f"\n🔍 Testing retrieval system...")
    
    for query in test_queries:
        try:
            embedding = embed_text(query)
            if embedding is None:
                continue
                
            results = index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            
            print(f"\nQuery: '{query}'")
            print(f"Results found: {len(results.matches)}")
            
            for i, match in enumerate(results.matches[:2]):
                source = Path(match.metadata.get('source', 'unknown')).name
                category = match.metadata.get('category', 'unknown')
                text_preview = match.metadata.get('text', '')[:100]
                print(f"  {i+1}. {source} ({category}) - Score: {match.score:.3f}")
                print(f"     Preview: {text_preview}...")
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

if __name__ == "__main__":
    try:
        processed_files, vector_count = ingest_documents()
        test_retrieval()
        
        print(f"\n✅ Ingestion successful!")
        print(f"Your AI companion now has access to {vector_count} knowledge chunks.")
        print(f"You can now interact with your assistant at: http://localhost:8000")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
