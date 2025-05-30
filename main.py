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

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
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
except Exception as e:
    raise Exception(f"Failed to create Pinecone index: {e}")


index = pc.Index(INDEX_NAME)

app = FastAPI()

# Simple in-memory conversation storage (upgrade to Redis/DB later)
conversation_memory = {}

# Initialize Google Calendar Manager
calendar_manager = GoogleCalendarManager()

# Simple session storage (upgrade to Redis/DB later)
user_sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

def embed(text: str) -> List[float]:
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def fetch_memories(q: str, k: int = 8) -> List[Dict]:
    """Fetch relevant memories with enhanced filtering and scoring"""
    results = index.query(
        vector=embed(q), 
        top_k=k, 
        include_metadata=True,
        filter={}
    ).matches
    
    # Filter by relevance score (only include high-confidence matches)
    filtered_results = [r for r in results if r.score > 0.7]
    return filtered_results if filtered_results else results[:3]

def format_memory_context(memories: List[Dict]) -> str:
    """Format memories into a coherent context"""
    if not memories:
        return ""

    context_parts = []
    categories = {}

    # Group memories by category
    for memory in memories:
        metadata = memory.metadata
        category = metadata.get('category', 'general')
        source = metadata.get('source', 'unknown')
        file_type = metadata.get('file_type', 'unknown')
        
        if category not in categories:
            categories[category] = []
        
        text_snippet = metadata.get('text', '')[:300]
        if len(metadata.get('text', '')) > 300:
            text_snippet += "..."
            
        categories[category].append({
            'text': text_snippet,
            'source': source,
            'file_type': file_type,
            'score': memory.score
        })

    # Format by category with more detail
    for category, items in categories.items():
        if category != 'general' and items:
            context_parts.append(f"\n{category.upper().replace('_', ' ')} DATA:")
            for item in items[:2]:  # Limit to top 2 per category
                source_name = os.path.basename(item['source'])
                context_parts.append(f"- From {source_name} ({item['file_type']}): {item['text']}")

    return "\n".join(context_parts)

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"Chat request: {req.message[:100]}...")  # Log incoming requests
        memories = fetch_memories(req.message)
        memory_context = format_memory_context(memories)

        # Get or create conversation history
        if req.session_id not in conversation_memory:
            conversation_memory[req.session_id] = []

        system_prompt = """You are Michael's helpful personal AI companion. You have access to Michael's notes, thoughts, and information.

Key traits:
- Be conversational and supportive
- Remember context from previous interactions
- Help with goals, projects, and daily tasks
- Provide insights based on stored memories
- Ask clarifying questions when helpful

Use the memories below to provide relevant, personalized assistance."""

        msgs = [{"role": "system", "content": system_prompt}]

        if memory_context:
            msgs.append({
                "role": "system", 
                "content": f"RELEVANT MEMORIES:\n{memory_context}"
            })

        # Add recent conversation history (last 6 messages)
        recent_conversation = conversation_memory[req.session_id][-6:]
        msgs.extend(recent_conversation)

        msgs.append({"role": "user", "content": req.message})

        resp = client.chat.completions.create(
            model="gpt-4o", 
            messages=msgs,
            temperature=0.7,
            max_tokens=500
        )
        
        response_content = resp.choices[0].message.content
        
        # Store conversation
        conversation_memory[req.session_id].append({"role": "user", "content": req.message})
        conversation_memory[req.session_id].append({"role": "assistant", "content": response_content})
        
        return {"response": response_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/status")
async def api_status():
    return {"message": "API is running"}

@app.get("/auth/google")
async def google_auth(request: Request):
    """Initiate Google OAuth flow"""
    redirect_uri = str(request.url_for('google_callback')).replace('http://', 'https://')
    flow = calendar_manager.create_auth_flow(redirect_uri)
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    # Store state in session (in production, use proper session management)
    user_sessions['oauth_state'] = state
    return RedirectResponse(authorization_url)

@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = None, state: str = None):
    """Handle Google OAuth callback"""
    if state != user_sessions.get('oauth_state'):
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    redirect_uri = str(request.url_for('google_callback')).replace('http://', 'https://')
    flow = calendar_manager.create_auth_flow(redirect_uri)
    
    authorization_response = str(request.url).replace('http://', 'https://')
    flow.fetch_token(authorization_response=authorization_response)
    
    # Store credentials (in production, encrypt and store securely)
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

@app.get("/api/upcoming-events")
async def get_upcoming_events():
    """Get upcoming calendar events"""
    if 'google_credentials' not in user_sessions:
        return {"events": [], "message": "Please connect your Google Calendar first", "auth_required": True}
    
    try:
        service = calendar_manager.get_calendar_service(user_sessions['google_credentials'])
        events = calendar_manager.get_upcoming_events(service)
        return {"events": events, "message": "Success"}
    except Exception as e:
        return {"events": [], "message": f"Error fetching events: {str(e)}"}

@app.post("/api/add-calendar-event")
async def add_calendar_event(event_data: dict):
    """Add calendar events via Google Calendar API"""
    if 'google_credentials' not in user_sessions:
        return {"message": "Please connect your Google Calendar first", "auth_required": True}
    
    try:
        service = calendar_manager.get_calendar_service(user_sessions['google_credentials'])
        
        # Format event for Google Calendar API
        google_event = {
            'summary': event_data.get('title', 'New Event'),
            'description': event_data.get('description', ''),
            'start': {
                'dateTime': event_data.get('start_time'),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': event_data.get('end_time'),
                'timeZone': 'UTC',
            },
        }
        
        if event_data.get('location'):
            google_event['location'] = event_data['location']
        
        event = calendar_manager.create_event(service, google_event)
        return {"message": "Event created successfully", "event_id": event.get('id') if event else None}
    except Exception as e:
        return {"message": f"Error creating event: {str(e)}"}

@app.post("/api/add-note")
async def add_quick_note(note: dict):
    """Add quick notes that get stored in memory"""
    try:
        note_text = note.get("text", "")
        category = note.get("category", "quick_notes")
        
        # Store in Pinecone memory
        from datetime import datetime
        vector_id = f"quick_note_{datetime.now().isoformat()}"
        metadata = {
            "text": note_text,
            "source": "quick_note",
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        index.upsert([(vector_id, embed(note_text), metadata)])
        
        return {"message": "Note added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding note: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")