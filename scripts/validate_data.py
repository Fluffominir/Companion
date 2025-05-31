
import os
import logging
from openai import OpenAI
from pinecone import Pinecone
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed_text(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def validate_system():
    """Comprehensive validation of the AI companion system"""
    
    print("🔍 AI COMPANION VALIDATION REPORT")
    print("=" * 50)
    
    # Check critical files
    print("\n📁 CHECKING SOURCE FILES:")
    source_dirs = ["docs/", "attached_assets/"]
    total_files = 0
    
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            files = list(Path(source_dir).rglob("*"))
            doc_files = [f for f in files if f.suffix.lower() in ['.pdf', '.md', '.txt']]
            print(f"  {source_dir}: {len(doc_files)} processable files")
            total_files += len(doc_files)
            
            # Show important files
            for file in doc_files[:5]:  # Show first 5
                size = file.stat().st_size
                print(f"    ✅ {file.name} ({size:,} bytes)")
        else:
            print(f"  ❌ {source_dir}: Directory not found")
    
    if total_files == 0:
        print("\n💥 CRITICAL: No source files found!")
        print("📋 Add your personal documents to docs/ or attached_assets/")
        return False
    
    # Check vector database
    print(f"\n🧠 CHECKING VECTOR DATABASE:")
    try:
        stats = index.describe_index_stats()
        print(f"  Total vectors: {stats.total_vector_count}")
        
        if stats.total_vector_count == 0:
            print("  ❌ Database is empty!")
            print("  💡 Run: python scripts/clean_ingest.py")
            return False
        else:
            print("  ✅ Database contains data")
            
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False
    
    # Test personal data queries
    print(f"\n🔎 TESTING PERSONAL DATA RETRIEVAL:")
    
    test_queries = [
        ("family information", "family, parents, relatives"),
        ("personality traits", "personality, characteristics, traits"),
        ("work details", "job, work, career, company"),
        ("goals and objectives", "goals, objectives, aspirations"),
        ("personal preferences", "likes, preferences, interests")
    ]
    
    successful_queries = 0
    
    for query_name, query_text in test_queries:
        print(f"\n   Testing: {query_name}")
        
        try:
            results = index.query(
                vector=embed_text(query_text),
                top_k=5,
                include_metadata=True
            ).matches
            
            relevant_results = [r for r in results if r.score > 0.3]
            
            if relevant_results:
                best = relevant_results[0]
                source = Path(best.metadata.get('source', 'unknown')).name
                category = best.metadata.get('category', 'unknown')
                print(f"     ✅ Found data: {source} (category: {category}, score: {best.score:.3f})")
                successful_queries += 1
            else:
                print(f"     ❌ No relevant data found")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Source files: {total_files}")
    print(f"   Vector count: {stats.total_vector_count}")
    print(f"   Successful queries: {successful_queries}/{len(test_queries)}")
    
    if successful_queries >= len(test_queries) // 2:
        print(f"\n✅ SYSTEM VALIDATION PASSED!")
        print(f"   Your AI companion should work well with current data.")
        return True
    else:
        print(f"\n⚠️  SYSTEM NEEDS IMPROVEMENT!")
        print(f"📋 RECOMMENDED ACTIONS:")
        print(f"   1. Add more personal documents to docs/")
        print(f"   2. Fill out docs/core_personal_data.md with your info")
        print(f"   3. Re-run: python scripts/clean_ingest.py")
        return False

def test_specific_query(query: str):
    """Test a specific query and show detailed results"""
    print(f"\n🔍 DETAILED QUERY TEST: '{query}'")
    print("-" * 40)
    
    try:
        results = index.query(
            vector=embed_text(query),
            top_k=5,
            include_metadata=True
        ).matches
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            source = Path(result.metadata.get('source', 'unknown')).name
            category = result.metadata.get('category', 'unknown')
            text_preview = result.metadata.get('text', '')[:150]
            
            print(f"{i}. Source: {source}")
            print(f"   Category: {category}")
            print(f"   Score: {result.score:.3f}")
            print(f"   Preview: {text_preview}...")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific query if provided
        query = " ".join(sys.argv[1:])
        test_specific_query(query)
    else:
        # Run full validation
        validate_system()
        
    print("\n🏁 Validation complete!")
