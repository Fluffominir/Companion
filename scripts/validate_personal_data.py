
import os
import sys
sys.path.append('..')
from main import index, embed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_personal_data():
    """Validate that personal data is properly stored and retrievable"""
    
    # Test queries for personal information
    test_queries = [
        "father name dad",
        "mother name mom", 
        "family members",
        "parents names",
        "family relationships",
        "Michael father",
        "Michael mother",
        "personality type",
        "personal background",
        "family background"
    ]
    
    print("🔍 VALIDATING PERSONAL DATA STORAGE...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔎 Testing query: '{query}'")
        
        try:
            # Search with multiple strategies
            results = index.query(
                vector=embed(query),
                top_k=20,
                include_metadata=True
            ).matches
            
            print(f"   Found {len(results)} total results")
            
            # Check for personal categories
            personal_results = [r for r in results if any(cat in r.metadata.get('category', '') 
                                                        for cat in ['personality', 'profile', 'personal', 'journal'])]
            
            print(f"   Found {len(personal_results)} personal category results")
            
            # Show top 3 results
            for i, result in enumerate(results[:3]):
                metadata = result.metadata
                print(f"   {i+1}. Score: {result.score:.3f} | Category: {metadata.get('category', 'unknown')}")
                print(f"      Source: {metadata.get('source', 'unknown')}")
                print(f"      Content preview: {metadata.get('text', '')[:150]}...")
                
                # Check for family-related keywords
                text_lower = metadata.get('text', '').lower()
                family_keywords = ['father', 'dad', 'mother', 'mom', 'parent', 'family', 'brother', 'sister']
                found_keywords = [kw for kw in family_keywords if kw in text_lower]
                if found_keywords:
                    print(f"      🏠 Family keywords found: {found_keywords}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check category distribution
    print(f"\n📊 CATEGORY DISTRIBUTION:")
    print("=" * 30)
    
    try:
        sample_results = index.query(
            vector=[0.0] * 1536,
            top_k=1000,
            include_metadata=True
        ).matches
        
        categories = {}
        sources = {}
        
        for result in sample_results:
            cat = result.metadata.get('category', 'unknown')
            source = result.metadata.get('source', 'unknown')
            
            categories[cat] = categories.get(cat, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        print("Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
            
        print("\nTop Sources:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {source}: {count}")
            
    except Exception as e:
        print(f"❌ Category analysis error: {e}")
    
    # Specific family data search
    print(f"\n👨‍👩‍👧‍👦 SPECIFIC FAMILY DATA SEARCH:")
    print("=" * 40)
    
    family_searches = [
        ("father father's dad daddy papa", "Father information"),
        ("mother mother's mom mommy mama", "Mother information"), 
        ("parents family background relatives", "General family information")
    ]
    
    for search_query, description in family_searches:
        print(f"\n🔍 {description}:")
        try:
            results = index.query(
                vector=embed(search_query),
                top_k=10,
                include_metadata=True
            ).matches
            
            found_family_info = False
            for result in results:
                text = result.metadata.get('text', '').lower()
                if any(word in text for word in ['father', 'dad', 'mother', 'mom', 'parent']):
                    found_family_info = True
                    print(f"  ✅ Found in {result.metadata.get('source', 'unknown')} (Score: {result.score:.3f})")
                    # Extract family-related sentences
                    sentences = result.metadata.get('text', '').split('.')
                    family_sentences = [s.strip() for s in sentences if any(word in s.lower() for word in ['father', 'dad', 'mother', 'mom', 'parent'])]
                    for sentence in family_sentences[:2]:  # Show first 2 relevant sentences
                        print(f"     → {sentence}")
            
            if not found_family_info:
                print(f"  ❌ No family information found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    validate_personal_data()
    print("\n🏁 Validation complete!")
    print("If family information is missing, run: python scripts/boot_memory.py")
    print("Make sure PDF files with personal data are in the docs/ folder")
