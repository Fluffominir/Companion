
import os
import logging
from openai import OpenAI
from pinecone import Pinecone
import json

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

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def validate_personal_data():
    """Validate that personal data is properly stored and retrievable"""
    
    print("🔍 PERSONAL DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Check if critical files exist
    critical_files = [
        "attached_assets/MichaelPersonality.pdf",
        "attached_assets/MichaelProfile.pdf",
        "docs/MichaelPersonality.pdf", 
        "docs/MichaelProfile.pdf"
    ]
    
    print("\n📁 CHECKING FOR CRITICAL FILES:")
    files_found = []
    for file_path in critical_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
            files_found.append(file_path)
        else:
            print(f"  ❌ {file_path} - NOT FOUND")
    
    if not files_found:
        print("\n💥 CRITICAL ISSUE: No personal files found!")
        print("📋 SOLUTION: Please ensure MichaelPersonality.pdf and MichaelProfile.pdf are uploaded")
        print("   - Place them in either docs/ or attached_assets/ folder")
        print("   - Then run: python scripts/boot_memory.py")
        return
    
    # Check what's in the vector database
    print(f"\n🧠 CHECKING VECTOR DATABASE:")
    try:
        stats = index.describe_index_stats()
        print(f"  Total vectors: {stats.total_vector_count}")
        
        if stats.total_vector_count == 0:
            print("  ❌ Database is empty! Run: python scripts/boot_memory.py")
            return
            
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return
    
    # Test personal data queries
    test_queries = [
        "family members",
        "mother name mom",
        "father name dad", 
        "parents names",
        "Michael family background",
        "personality type",
        "personal information"
    ]
    
    print(f"\n🔎 TESTING PERSONAL DATA RETRIEVAL:")
    
    total_good_results = 0
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        try:
            # Search all categories
            all_results = index.query(
                vector=embed(query),
                top_k=20,
                include_metadata=True
            ).matches
            
            # Search personal categories specifically
            personal_results = index.query(
                vector=embed(query),
                top_k=10,
                include_metadata=True,
                filter={"category": {"$in": ["personality_core", "profile_core", "personality", "personal_database"]}}
            ).matches
            
            print(f"     All results: {len(all_results)}")
            print(f"     Personal results: {len(personal_results)}")
            
            # Show best results
            relevant_all = [r for r in all_results if r.score > 0.25]
            relevant_personal = [r for r in personal_results if r.score > 0.2]
            
            if relevant_personal:
                best = relevant_personal[0]
                source = best.metadata.get('source', 'unknown')
                category = best.metadata.get('category', 'unknown')
                text_preview = best.metadata.get('text', '')[:100]
                print(f"     ✅ Best personal match: {source} (score: {best.score:.3f})")
                print(f"        Category: {category}")
                print(f"        Preview: {text_preview}...")
                total_good_results += 1
            elif relevant_all:
                best = relevant_all[0]
                source = best.metadata.get('source', 'unknown')
                category = best.metadata.get('category', 'unknown')
                print(f"     ⚠️  Best match (non-personal): {source} (score: {best.score:.3f})")
                print(f"        Category: {category}")
            else:
                print(f"     ❌ No relevant results found")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Summary and recommendations
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Files found: {len(files_found)}")
    print(f"   Queries with good results: {total_good_results}/{len(test_queries)}")
    
    if total_good_results < len(test_queries) // 2:
        print(f"\n⚠️  DATA ACCESS ISSUES DETECTED!")
        print(f"📋 RECOMMENDED ACTIONS:")
        print(f"   1. Re-run memory processing: python scripts/boot_memory.py")
        print(f"   2. Check if PDFs contain readable text (not just images)")
        print(f"   3. Verify PDF files aren't corrupted")
        print(f"   4. Consider adding specific family information to a markdown file in docs/")
    else:
        print(f"\n✅ PERSONAL DATA ACCESS LOOKS GOOD!")
        print(f"   The AI should be able to access your personal information.")
    
    # Test a specific family query
    print(f"\n🔍 SPECIFIC FAMILY INFORMATION TEST:")
    family_query = "mother father parent family member name"
    try:
        results = index.query(
            vector=embed(family_query),
            top_k=10,
            include_metadata=True
        ).matches
        
        family_mentions = []
        for result in results:
            text = result.metadata.get('text', '').lower()
            if any(word in text for word in ['mother', 'father', 'mom', 'dad', 'parent', 'family']):
                family_mentions.append({
                    'score': result.score,
                    'source': result.metadata.get('source', ''),
                    'text_snippet': result.metadata.get('text', '')[:200]
                })
        
        if family_mentions:
            print(f"   Found {len(family_mentions)} potential family mentions:")
            for i, mention in enumerate(family_mentions[:3]):
                print(f"   {i+1}. {mention['source']} (score: {mention['score']:.3f})")
                print(f"      {mention['text_snippet'][:150]}...")
        else:
            print(f"   ❌ No family information found in any documents!")
            print(f"   💡 This explains why the AI can't answer family questions.")
            
    except Exception as e:
        print(f"   ❌ Error testing family data: {e}")

if __name__ == "__main__":
    validate_personal_data()
    print("\n🏁 Validation complete!")
    print("If issues persist, check that your PDFs contain actual text and aren't just scanned images.")
