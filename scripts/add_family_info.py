
import os
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def add_family_information():
    """Emergency function to add family information directly"""
    
    print("🏠 ADDING FAMILY INFORMATION TO DATABASE")
    print("Please enter your family information below:")
    print("(Press Enter to skip any field)")
    
    # Collect family information
    mother_name = input("Mother's name: ").strip()
    father_name = input("Father's name: ").strip()
    
    family_info = []
    
    if mother_name:
        family_info.append(f"Michael's mother's name is {mother_name}.")
        
    if father_name:
        family_info.append(f"Michael's father's name is {father_name}.")
    
    # Add any additional family info
    additional_info = input("Any other family information (siblings, spouse, etc.): ").strip()
    if additional_info:
        family_info.append(additional_info)
    
    if not family_info:
        print("No family information provided. Exiting.")
        return
    
    # Store in vector database
    combined_info = " ".join(family_info)
    
    vector_id = f"family_info_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = {
        "text": combined_info,
        "source": "manual_family_entry",
        "category": "personal_database",
        "priority": "highest",
        "timestamp": datetime.now().isoformat(),
        "file_type": "manual_entry"
    }
    
    try:
        embedding = embed(combined_info)
        index.upsert([(vector_id, embedding, metadata)])
        
        print(f"\n✅ Successfully added family information:")
        print(f"   {combined_info}")
        print(f"\n🧠 This information is now available to your AI companion!")
        
    except Exception as e:
        print(f"❌ Error adding family information: {e}")

if __name__ == "__main__":
    add_family_information()
