
import os
from pinecone import Pinecone

# Initialize
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("companion-memory")

try:
    # Check index stats
    stats = index.describe_index_stats()
    print(f"✅ Index is accessible")
    print(f"   Total vectors: {stats.total_vector_count}")
    print(f"   Index status: Ready")
    
    # Try a simple query
    test_query = index.query(
        vector=[0.0] * 1536,
        top_k=1,
        include_metadata=True
    )
    print(f"✅ Query test successful")
    
except Exception as e:
    print(f"❌ Index error: {e}")
    print("This might indicate the index is locked or having issues")
