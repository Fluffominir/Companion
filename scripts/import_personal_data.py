
import os
import json
import requests
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

def process_notion_data():
    """Process Notion links and create memory entries"""
    notion_data = [
        {
            "title": "Michael's Personal Database",
            "url": "https://www.notion.so/michaelslusher/fe686d01a20545339a92b25832875fd5",
            "category": "personal_database",
            "description": "Comprehensive personal information and data tracking"
        },
        {
            "title": "Rocket Launch Studio Dashboard", 
            "url": "https://www.notion.so/Rocket-Launch-Studio-Dashboard-1859de66e9048000b6e2e6d6c0195490",
            "category": "work",
            "description": "Professional dashboard and company information"
        }
    ]
    
    for item in notion_data:
        vector_id = f"notion_{item['category']}_{datetime.now().isoformat()}"
        text_content = f"{item['title']}: {item['description']} (Reference: {item['url']})"
        
        metadata = {
            "text": text_content,
            "source": "notion_link",
            "category": item["category"],
            "url": item["url"],
            "title": item["title"],
            "timestamp": datetime.now().isoformat()
        }
        
        index.upsert([(vector_id, embed(text_content), metadata)])
        print(f"Added Notion reference: {item['title']}")

def add_personal_context():
    """Add personal context based on file names you mentioned"""
    personal_contexts = [
        {
            "category": "personality",
            "content": "Personality assessment data and type information for understanding communication and work preferences"
        },
        {
            "category": "journals",
            "content": "Personal journals from 2022, 2024, and 2025 containing thoughts, reflections, and personal growth insights"
        },
        {
            "category": "health",
            "content": "Health records and medical information for wellness tracking and health-related decision making"
        },
        {
            "category": "work",
            "content": "Rocket Launch Studio company information, employee handbook, and professional context"
        },
        {
            "category": "books",
            "content": "Key books read: Attached (attachment theory), The Body Keeps the Score (trauma), E-Myth Revisited (business), Show Your Work and Steal Like an Artist (creativity)"
        },
        {
            "category": "projects",
            "content": "SERWM pitch deck and brand documentation, various creative and business projects"
        }
    ]
    
    for item in personal_contexts:
        vector_id = f"context_{item['category']}_{datetime.now().isoformat()}"
        
        metadata = {
            "text": item["content"],
            "source": "personal_context",
            "category": item["category"],
            "timestamp": datetime.now().isoformat()
        }
        
        index.upsert([(vector_id, embed(item["content"]), metadata)])
        print(f"Added personal context: {item['category']}")

if __name__ == "__main__":
    print("Processing personal data...")
    process_notion_data()
    add_personal_context()
    print("Personal data import complete!")
