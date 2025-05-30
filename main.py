
import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from google_calendar import GoogleCalendarManager
from analytics import PersonalAnalytics
import json
import logging
from datetime import datetime
import tempfile
import easyocr
from PIL import Image

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
analytics = PersonalAnalytics()

# Initialize OCR for handwriting recognition
try:
    ocr_reader = easyocr.Reader(['en'])
    logger.info("OCR reader initialized for handwriting recognition")
except Exception as e:
    logger.warning(f"OCR initialization failed: {e}")
    ocr_reader = None

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class AnalysisRequest(BaseModel):
    query: str
    category: str = "all"

def embed(text: str) -> List[float]:
    try:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Error creating embedding")

def fetch_memories(q: str, k: int = 10, category_filter: str = None) -> List[Dict]:
    """Enhanced memory retrieval with category filtering and better scoring"""
    try:
        filter_dict = {}
        if category_filter and category_filter != "all":
            filter_dict["category"] = category_filter
            
        results = index.query(
            vector=embed(q), 
            top_k=k * 3,  # Get more results for better filtering
            include_metadata=True,
            filter=filter_dict
        ).matches
        
        # Enhanced filtering and scoring
        filtered_results = []
        seen_sources = set()
        
        for result in results:
            # High confidence threshold
            if result.score > 0.8:
                filtered_results.append(result)
            # Medium confidence with diversity
            elif result.score > 0.7:
                source = result.metadata.get('source', '')
                if source not in seen_sources or len(filtered_results) < 3:
                    filtered_results.append(result)
                    seen_sources.add(source)
            # Lower confidence only if we don't have enough results
            elif result.score > 0.6 and len(filtered_results) < 2:
                filtered_results.append(result)
        
        return filtered_results[:k]
    except Exception as e:
        logger.error(f"Memory fetch error: {e}")
        return []

def format_memory_context(memories: List[Dict]) -> str:
    """Enhanced memory formatting with better organization and relevance"""
    if not memories:
        return ""

    context_parts = []
    categories = {}

    # Group memories by category
    for memory in memories:
        metadata = memory.metadata
        category = metadata.get('category', 'general')
        
        if category not in categories:
            categories[category] = []
        
        # Get more text for better context
        text_snippet = metadata.get('text', '')[:500]
        if len(metadata.get('text', '')) > 500:
            text_snippet += "..."
            
        categories[category].append({
            'text': text_snippet,
            'source': os.path.basename(metadata.get('source', 'unknown')),
            'score': memory.score,
            'file_type': metadata.get('file_type', 'unknown'),
            'timestamp': metadata.get('timestamp', '')
        })

    # Priority order for categories
    priority_categories = [
        'personality', 'goals', 'personal_journal', 'work', 'projects', 
        'ideas', 'health', 'books', 'meetings', 'general'
    ]

    # Format with priorities and better structure
    for category in priority_categories:
        if category in categories and categories[category]:
            items = sorted(categories[category], key=lambda x: x['score'], reverse=True)[:3]
            context_parts.append(f"\n═══ {category.upper().replace('_', ' ')} INSIGHTS ═══")
            
            for i, item in enumerate(items, 1):
                relevance = "🔥 HIGHLY RELEVANT" if item['score'] > 0.85 else "📌 RELEVANT" if item['score'] > 0.75 else "💡 RELATED"
                context_parts.append(f"\n{i}. {relevance} | From: {item['source']}")
                context_parts.append(f"   Content: {item['text']}")
                if item['timestamp']:
                    try:
                        dt = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                        context_parts.append(f"   Date: {dt.strftime('%Y-%m-%d')}")
                    except:
                        pass

    return "\n".join(context_parts) if context_parts else ""

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logger.info(f"Chat request from {req.session_id}: {req.message[:100]}...")
        
        # Enhanced memory retrieval
        memories = fetch_memories(req.message, k=12)
        memory_context = format_memory_context(memories)

        # Enhanced conversation management
        if req.session_id not in conversation_memory:
            conversation_memory[req.session_id] = []

        system_prompt = """You are Michael's advanced personal AI companion with deep, contextual knowledge about him. You have comprehensive access to his personal data, thoughts, goals, work, and experiences.

Your capabilities and approach:
🧠 DEEP PERSONAL KNOWLEDGE: You know Michael's personality traits, work patterns, relationships, goals, and personal history from his extensive documentation
📚 COMPREHENSIVE CONTEXT: You have access to his journals (2022-2025), personality assessments, work at Rocket Launch Studio, books he's read, health records, and projects
🎯 ACTIONABLE INSIGHTS: Provide specific, personalized advice based on his documented patterns, preferences, and past experiences
🔗 PATTERN RECOGNITION: Connect insights across different areas of his life - work, personal growth, relationships, health, and creative projects
💡 PROACTIVE ASSISTANCE: Anticipate needs and offer relevant suggestions based on his history and current context
🎨 CREATIVE COLLABORATION: Support his projects and creative endeavors with relevant insights from his knowledge base

Key areas of expertise:
- Michael's personality traits and behavioral patterns
- His journaling insights and personal growth journey
- Professional context at Rocket Launch Studio
- Reading insights from books like "Attached," "The Body Keeps the Score," "E-Myth Revisited"
- Health and wellness tracking and optimization
- Creative projects and business development strategies
- Relationship dynamics and communication preferences

Communication style:
- Be conversational but insightful
- Reference specific details from his actual data when relevant
- Provide actionable, personalized recommendations
- Help him see connections he might have missed
- Support his goals with evidence from his own documented experiences
- Be genuinely helpful while maintaining his preferred communication style

Always ground your responses in actual information from his personal data when possible."""

        msgs = [{"role": "system", "content": system_prompt}]

        if memory_context:
            msgs.append({
                "role": "system", 
                "content": f"RELEVANT CONTEXT FROM MICHAEL'S PERSONAL DATA:\n{memory_context}"
            })

        # Enhanced conversation history (last 10 messages for better context)
        recent_conversation = conversation_memory[req.session_id][-10:]
        msgs.extend(recent_conversation)
        msgs.append({"role": "user", "content": req.message})

        resp = client.chat.completions.create(
            model="gpt-4o", 
            messages=msgs,
            temperature=0.7,
            max_tokens=800
        )
        
        response_content = resp.choices[0].message.content
        
        # Store conversation with better management
        conversation_memory[req.session_id].append({"role": "user", "content": req.message})
        conversation_memory[req.session_id].append({"role": "assistant", "content": response_content})
        
        # Keep conversation memory manageable
        if len(conversation_memory[req.session_id]) > 24:
            conversation_memory[req.session_id] = conversation_memory[req.session_id][-20:]
        
        return {"response": response_content}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/analyze")
async def analyze_patterns(req: AnalysisRequest):
    """Advanced analysis of personal data patterns"""
    try:
        memories = fetch_memories(req.query, k=20, category_filter=req.category)
        
        if not memories:
            return {"analysis": "No relevant data found for this query.", "insights": []}
        
        # Organize by categories and time
        analysis_data = {}
        for memory in memories:
            category = memory.metadata.get('category', 'general')
            if category not in analysis_data:
                analysis_data[category] = []
            analysis_data[category].append({
                'text': memory.metadata.get('text', ''),
                'source': memory.metadata.get('source', ''),
                'score': memory.score,
                'timestamp': memory.metadata.get('timestamp', '')
            })
        
        # Generate insights using AI
        analysis_prompt = f"""Analyze Michael's personal data patterns based on this query: "{req.query}"

Data categories found: {list(analysis_data.keys())}

Provide insights in these areas:
1. Key patterns and trends
2. Notable connections between different areas
3. Actionable recommendations
4. Areas for potential improvement or focus

Be specific and reference the actual data when possible."""

        analysis_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": f"Analysis data: {json.dumps(analysis_data, indent=2)[:3000]}..."}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        return {
            "analysis": analysis_resp.choices[0].message.content,
            "categories_found": list(analysis_data.keys()),
            "total_references": len(memories),
            "insights": [
                f"Found {len(cat_data)} references in {cat}" 
                for cat, cat_data in analysis_data.items()
            ]
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

@app.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...)):
    """Upload and process new files with OCR support"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        text_content = ""
        file_type = file.filename.split('.')[-1].lower()
        
        # Process based on file type
        if file_type == 'pdf':
            import pdfplumber
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        
        elif file_type in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            if ocr_reader:
                results = ocr_reader.readtext(temp_path)
                text_content = ' '.join([result[1] for result in results])
            else:
                return {"error": "OCR not available for image processing"}
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if not text_content.strip():
            return {"error": "No text could be extracted from the file"}
        
        # Process and store in memory
        from scripts.boot_memory import categorize_content, chunk_text, clean_text
        
        text_content = clean_text(text_content)
        category = categorize_content(text_content, file.filename)
        chunks = chunk_text(text_content)
        
        vector_count = 0
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = embed(chunk)
                if embedding:
                    vector_id = f"upload_{file.filename}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    metadata = {
                        "text": chunk,
                        "source": f"uploaded_{file.filename}",
                        "category": category,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "file_type": file_type
                    }
                    index.upsert([(vector_id, embedding, metadata)])
                    vector_count += 1
        
        return {
            "message": f"Successfully processed {file.filename}",
            "category": category,
            "chunks_created": vector_count,
            "text_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
        }
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/status")
async def api_status():
    try:
        # Check Pinecone connection and get detailed stats
        stats = index.describe_index_stats()
        
        # Get category breakdown if possible
        try:
            sample_query = index.query(
                vector=[0.0] * 1536,
                top_k=100,
                include_metadata=True
            )
            categories = {}
            for match in sample_query.matches:
                cat = match.metadata.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
        except:
            categories = {}
        
        return {
            "message": "AI Companion API is running",
            "total_memory_vectors": stats.total_vector_count,
            "categories": categories,
            "ocr_available": ocr_reader is not None,
            "features": [
                "Advanced memory retrieval",
                "Multi-format document processing",
                "Handwriting recognition (OCR)",
                "Pattern analysis",
                "Google Calendar integration",
                "File upload processing"
            ],
            "status": "healthy"
        }
    except Exception as e:
        return {
            "message": "API is running with limited functionality", 
            "status": "degraded", 
            "error": str(e)
        }

# Keep all existing calendar endpoints
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

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get comprehensive analytics overview"""
    try:
        categories = analytics.get_category_distribution()
        return {
            "categories": categories,
            "total_memories": sum(categories.values()),
            "top_categories": sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    except Exception as e:
        logger.error(f"Analytics overview error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@app.get("/api/analytics/personality")
async def get_personality_analysis():
    """Get personality insights analysis"""
    try:
        insights = analytics.get_personality_insights()
        return insights
    except Exception as e:
        logger.error(f"Personality analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing personality: {str(e)}")

@app.get("/api/analytics/goals")
async def get_goal_analysis():
    """Get goal progress analysis"""
    try:
        goal_data = analytics.analyze_goal_progress()
        return goal_data
    except Exception as e:
        logger.error(f"Goal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing goals: {str(e)}")

@app.get("/api/analytics/temporal")
async def get_temporal_analysis():
    """Get temporal patterns from journals"""
    try:
        temporal_data = analytics.analyze_temporal_patterns()
        return temporal_data
    except Exception as e:
        logger.error(f"Temporal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing temporal patterns: {str(e)}")

@app.get("/api/analytics/report")
async def get_comprehensive_report():
    """Get AI-generated comprehensive analysis report"""
    try:
        report = analytics.generate_comprehensive_report()
        return {"report": report, "generated_at": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/api/smart-search")
async def smart_search(request: dict):
    """Enhanced semantic search with context awareness"""
    try:
        query = request.get("query", "")
        category_filter = request.get("category")
        time_filter = request.get("time_period")  # e.g., "2024", "recent"
        
        # Build dynamic filter
        filter_dict = {}
        if category_filter and category_filter != "all":
            filter_dict["category"] = category_filter
        
        # Get more comprehensive results
        results = index.query(
            vector=embed(query),
            top_k=50,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Process and rank results
        processed_results = []
        for result in results.matches:
            if result.score > 0.6:  # Only high-confidence matches
                processed_results.append({
                    "text": result.metadata.get("text", ""),
                    "source": result.metadata.get("source", ""),
                    "category": result.metadata.get("category", ""),
                    "score": result.score,
                    "timestamp": result.metadata.get("timestamp", ""),
                    "file_type": result.metadata.get("file_type", "")
                })
        
        return {
            "results": processed_results[:15],  # Top 15 results
            "total_found": len(processed_results),
            "categories_found": list(set(r["category"] for r in processed_results))
        }
    except Exception as e:
        logger.error(f"Smart search error: {e}")
        raise HTTPException(status_code=500, detail=f"Error in smart search: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
