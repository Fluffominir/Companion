
import os
import sys
sys.path.append('..')
from boot_memory import *

def process_priority_files():
    """Focus on processing Michael's most important personal data"""
    priority_files = [
        "docs/MichaelPersonality.pdf",
        "docs/MichaelProfile.pdf", 
        "attached_assets/MichaelPersonality.pdf",
        "attached_assets/MichaelProfile.pdf",
        "docs/Journal 2022.pdf",
        "docs/2024.pdf", 
        "docs/2025 Journal.pdf"
    ]
    
    print("🎯 Processing PRIORITY personal data files...")
    
    for file_path in priority_files:
        if os.path.exists(file_path):
            print(f"📄 Processing: {file_path}")
            try:
                if file_path.endswith('.pdf'):
                    text = extract_pdf_text_advanced(file_path)
                    if text:
                        # Use higher chunk overlap for personality data
                        chunks = chunk_text(text, max_length=600)
                        category = "personality_core" if "personality" in file_path.lower() else "profile_core"
                        
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                embedding = embed(chunk)
                                if embedding:
                                    vector_id = f"priority_{os.path.basename(file_path)}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    metadata = {
                                        "text": chunk,
                                        "source": file_path,
                                        "category": category,
                                        "priority": "high",
                                        "chunk_index": i,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    index.upsert([(vector_id, embedding, metadata)])
                        print(f"✅ Created {len(chunks)} priority vectors for {file_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

    # Add Notion page references with high priority
    notion_references = [
        {
            "title": "Michael's Personal Database - Core Profile",
            "url": "https://www.notion.so/michaelslusher/fe686d01a20545339a92b25832875fd5",
            "content": "CORE PERSONAL DATA: Family information, personal details, preferences, relationships, background - contains mother's name and family details"
        }
    ]
    
    for notion in notion_references:
        embedding = embed(notion["content"])
        if embedding:
            vector_id = f"notion_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = {
                "text": notion["content"],
                "source": notion["url"], 
                "category": "personal_core",
                "priority": "highest",
                "title": notion["title"],
                "timestamp": datetime.now().isoformat()
            }
            index.upsert([(vector_id, embedding, metadata)])
            print(f"✅ Added high-priority Notion reference: {notion['title']}")

if __name__ == "__main__":
    process_priority_files()
    print("🎯 Priority data processing complete!")
