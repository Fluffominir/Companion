
import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from google_calendar import GoogleCalendarManager
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set")

INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

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
    raise Exception(f"Failed to create Pinecone index: {e}")

app = FastAPI()

# Enhanced conversation storage
conversation_memory = {}
calendar_manager = GoogleCalendarManager()
user_sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

def embed(text: str) -> List[float]:
    try:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Error creating embedding")

def fetch_memories(q: str, k: int = 8) -> List[Dict]:
    """Enhanced memory retrieval with better filtering"""
    try:
        results = index.query(
            vector=embed(q), 
            top_k=k * 2,  # Get more results to filter better
            include_metadata=True,
            filter={}
        ).matches
        
        # Enhanced filtering based on relevance and diversity
        filtered_results = []
        seen_sources = set()
        
        for result in results:
            if result.score > 0.75:  # High confidence
                filtered_results.append(result)
            elif result.score > 0.65 and len(filtered_results) < 3:  # Medium confidence
                source = result.metadata.get('source', '')
                if source not in seen_sources:
                    filtered_results.append(result)
                    seen_sources.add(source)
        
        return filtered_results[:k]
    except Exception as e:
        logger.error(f"Memory fetch error: {e}")
        return []

def format_memory_context(memories: List[Dict]) -> str:
    """Enhanced memory formatting with better organization"""
    if not memories:
        return ""

    context_parts = []
    categories = {}

    # Group and prioritize memories
    for memory in memories:
        metadata = memory.metadata
        category = metadata.get('category', 'general')
        
        if category not in categories:
            categories[category] = []
        
        text_snippet = metadata.get('text', '')[:400]
        if len(metadata.get('text', '')) > 400:
            text_snippet += "..."
            
        categories[category].append({
            'text': text_snippet,
            'source': os.path.basename(metadata.get('source', 'unknown')),
            'score': memory.score,
            'file_type': metadata.get('file_type', 'unknown')
        })

    # Priority order for categories
    priority_categories = [
        'personality', 'goals', 'work', 'projects', 
        'personal_journal', 'ideas', 'health', 'books'
    ]

    # Format with priorities
    for category in priority_categories:
        if category in categories and categories[category]:
            items = sorted(categories[category], key=lambda x: x['score'], reverse=True)[:2]
            context_parts.append(f"\n{category.upper().replace('_', ' ')} CONTEXT:")
            for item in items:
                context_parts.append(f"- From {item['source']}: {item['text']}")

    return "\n".join(context_parts)

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logger.info(f"Chat request from {req.session_id}: {req.message[:100]}...")
        
        memories = fetch_memories(req.message)
        memory_context = format_memory_context(memories)

        # Enhanced conversation management
        if req.session_id not in conversation_memory:
            conversation_memory[req.session_id] = []

        system_prompt = """You are Michael's personal AI companion and assistant. You have deep knowledge about Michael from his journals, goals, personality assessments, work at Rocket Launch Studio, books he's read, and personal preferences.

Core traits and approach:
- Be genuinely helpful and insightful, drawing from Michael's actual data
- Reference specific details from his journals, goals, and experiences when relevant
- Maintain context about his work, relationships, and personal growth journey
- Provide actionable advice based on his documented patterns and preferences
- Be conversational but professional, matching his communication style
- Help him connect ideas across different areas of his life
- Support his goals at Rocket Launch Studio and personal projects

Key areas of knowledge:
- Michael's personality traits and assessment results
- His journaling patterns and insights from 2022-2025
- Goals and aspirations 
- Work context at Rocket Launch Studio
- Reading and learning from books like "Attached," "The Body Keeps the Score," etc.
- Health and wellness tracking
- Creative projects and business development

Always be specific and reference actual information when possible."""

        msgs = [{"role": "system", "content": system_prompt}]

        if memory_context:
            msgs.append({
                "role": "system", 
                "content": f"RELEVANT CONTEXT FROM MICHAEL'S DATA:\n{memory_context}"
            })

        # Enhanced conversation history (last 8 messages)
        recent_conversation = conversation_memory[req.session_id][-8:]
        msgs.extend(recent_conversation)
        msgs.append({"role": "user", "content": req.message})

        resp = client.chat.completions.create(
            model="gpt-4o", 
            messages=msgs,
            temperature=0.7,
            max_tokens=600
        )
        
        response_content = resp.choices[0].message.content
        
        # Store conversation with better management
        conversation_memory[req.session_id].append({"role": "user", "content": req.message})
        conversation_memory[req.session_id].append({"role": "assistant", "content": response_content})
        
        # Keep conversation memory manageable
        if len(conversation_memory[req.session_id]) > 20:
            conversation_memory[req.session_id] = conversation_memory[req.session_id][-16:]
        
        return {"response": response_content}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/status")
async def api_status():
    try:
        # Check Pinecone connection
        stats = index.describe_index_stats()
        return {
            "message": "API is running",
            "memory_vectors": stats.total_vector_count,
            "status": "healthy"
        }
    except Exception as e:
        return {"message": "API is running", "status": "degraded", "error": str(e)}

@app.get("/auth/google")
async def google_auth(request: Request):
    """Initiate Google OAuth flow"""
    try:
        redirect_uri = str(request.url_for('google_callback')).replace('http://', 'https://')
        flow = calendar_manager.create_auth_flow(redirect_uri)
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        user_sessions['oauth_state'] = state
        return RedirectResponse(authorization_url)
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication error")

@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = None, state: str = None):
    """Handle Google OAuth callback"""
    try:
        if state != user_sessions.get('oauth_state'):
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        redirect_uri = str(request.url_for('google_callback')).replace('http://', 'https://')
        flow = calendar_manager.create_auth_flow(redirect_uri)
        
        authorization_response = str(request.url).replace('http://', 'https://')
        flow.fetch_token(authorization_response=authorization_response)
        
        credentials = flow.credentials
        user_sessions['google_credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        return RedirectResponse("/", status_code=302)
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse("/?error=auth_failed", status_code=302)

@app.get("/api/upcoming-events")
async def get_upcoming_events():
    """Get upcoming calendar events"""
    if 'google_credentials' not in user_sessions:
        return {"events": [], "message": "Please connect your Google Calendar first", "auth_required": True}
    
    try:
        service = calendar_manager.get_calendar_service(user_sessions['google_credentials'])
        events = calendar_manager.get_upcoming_events(service, max_results=10)
        return {"events": events, "message": "Success", "auth_required": False}
    except Exception as e:
        logger.error(f"Calendar events error: {e}")
        return {"events": [], "message": f"Error fetching events: {str(e)}", "auth_required": False}

@app.post("/api/add-calendar-event")
async def add_calendar_event(event_data: dict):
    """Add calendar events via Google Calendar API"""
    if 'google_credentials' not in user_sessions:
        return {"message": "Please connect your Google Calendar first", "auth_required": True}
    
    try:
        service = calendar_manager.get_calendar_service(user_sessions['google_credentials'])
        
        google_event = {
            'summary': event_data.get('title', 'New Event'),
            'description': event_data.get('description', ''),
            'start': {
                'dateTime': event_data.get('start_time'),
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': event_data.get('end_time'),
                'timeZone': 'America/New_York',
            },
        }
        
        if event_data.get('location'):
            google_event['location'] = event_data['location']
        
        event = calendar_manager.create_event(service, google_event)
        return {"message": "Event created successfully", "event_id": event.get('id') if event else None}
    except Exception as e:
        logger.error(f"Create event error: {e}")
        return {"message": f"Error creating event: {str(e)}"}

@app.post("/api/add-note")
async def add_quick_note(note: dict):
    """Add quick notes that get stored in memory"""
    try:
        note_text = note.get("text", "")
        category = note.get("category", "quick_notes")
        
        from datetime import datetime
        vector_id = f"quick_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            "text": note_text,
            "source": "quick_note",
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "file_type": "note"
        }
        index.upsert([(vector_id, embed(note_text), metadata)])
        
        return {"message": "Note added successfully"}
    except Exception as e:
        logger.error(f"Add note error: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding note: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
