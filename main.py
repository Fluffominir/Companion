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
from integrations import IntegrationsManager
from voice_handler import VoiceHandler
from daily_insights import DailyInsightsManager
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
integrations = IntegrationsManager()
voice_handler = VoiceHandler()
insights_manager = DailyInsightsManager()

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

def fetch_memories(q: str, k: int = 15, category_filter: str = None) -> List[Dict]:
    """Enhanced memory retrieval with improved filtering for personal data accuracy"""
    try:
        logger.info(f"Searching for: '{q}' with category filter: {category_filter}")

        # Multiple search strategies for comprehensive retrieval
        all_results = []
        
        # Strategy 1: Direct semantic search
        filter_dict = {}
        if category_filter and category_filter != "all":
            filter_dict["category"] = category_filter

        semantic_results = index.query(
            vector=embed(q), 
            top_k=k * 3,
            include_metadata=True,
            filter=filter_dict
        ).matches
        all_results.extend(semantic_results)

        # Strategy 2: Search in high-priority personal categories if no filter specified
        if not category_filter:
            priority_categories = ['personality', 'personality_core', 'profile_core', 'personal_journal', 
                                 'journal_2024', 'journal_2025', 'journal_2022', 'personal_database']
            
            for category in priority_categories:
                try:
                    priority_results = index.query(
                        vector=embed(q),
                        top_k=10,
                        include_metadata=True,
                        filter={"category": category}
                    ).matches
                    all_results.extend(priority_results)
                except Exception as e:
                    logger.warning(f"Error searching category {category}: {e}")

        logger.info(f"Found {len(all_results)} total results from all strategies")

        # Enhanced filtering with lower thresholds for personal data
        filtered_results = []
        seen_content = set()

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        for result in all_results:
            if len(filtered_results) >= k:
                break
                
            content_preview = result.metadata.get('text', '')[:150]
            category = result.metadata.get('category', '')
            source = result.metadata.get('source', '')
            
            # Skip duplicates
            if content_preview in seen_content:
                continue
                
            # More inclusive thresholds based on content type
            min_score = 0.3  # Base threshold
            
            # Lower thresholds for personal/family content
            personal_keywords = ['family', 'father', 'mother', 'dad', 'mom', 'parent', 'brother', 'sister', 
                               'wife', 'husband', 'name', 'born', 'grew up', 'childhood', 'relationship']
            if any(keyword.lower() in q.lower() for keyword in personal_keywords):
                min_score = 0.25  # More inclusive for personal queries
                
            # Even lower for high-priority categories
            if category in ['personality', 'personality_core', 'profile_core', 'personal_database']:
                min_score = 0.2
                
            # Priority boost for core personal files
            if any(priority_file in source.lower() for priority_file in ['personality', 'profile', 'personal']):
                min_score = 0.15

            if result.score >= min_score:
                filtered_results.append(result)
                seen_content.add(content_preview)
                logger.info(f"INCLUDED (score: {result.score:.3f}, category: {category}): {source}")

        logger.info(f"Returning {len(filtered_results)} filtered results")
        return filtered_results

    except Exception as e:
        logger.error(f"Memory fetch error: {e}")
        return []

def format_memory_context(memories: List[Dict]) -> str:
    """Enhanced memory formatting with better organization for personal data accuracy"""
    if not memories:
        return ""

    context_parts = []
    categories = {}

    # Group memories by category with deduplication
    seen_content = set()
    for memory in memories:
        metadata = memory.metadata
        category = metadata.get('category', 'general')
        text_content = metadata.get('text', '')
        
        # Skip near-duplicates
        content_hash = text_content[:200]
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        if category not in categories:
            categories[category] = []

        # Get more text for personal queries, especially family information
        text_snippet = text_content[:800]  # Increased for personal context
        if len(text_content) > 800:
            text_snippet += "..."

        categories[category].append({
            'text': text_snippet,
            'source': os.path.basename(metadata.get('source', 'unknown')),
            'score': memory.score,
            'file_type': metadata.get('file_type', 'unknown'),
            'timestamp': metadata.get('timestamp', ''),
            'category': category
        })

    # Priority order for categories - personal data first
    priority_categories = [
        'personality_core', 'profile_core', 'personality', 'personal_database',
        'personal_journal', 'journal_2025', 'journal_2024', 'journal_2022', 
        'family', 'relationships', 'goals', 'work', 'projects', 'health', 
        'books', 'ideas', 'meetings', 'general'
    ]

    # Add any categories not in priority list
    all_categories = set(categories.keys())
    for cat in all_categories:
        if cat not in priority_categories:
            priority_categories.append(cat)

    # Format with priorities and better structure
    for category in priority_categories:
        if category in categories and categories[category]:
            items = sorted(categories[category], key=lambda x: x['score'], reverse=True)
            
            # Show more items for personal categories
            max_items = 5 if any(personal_cat in category for personal_cat in 
                               ['personality', 'personal', 'profile', 'family', 'journal']) else 3
            items = items[:max_items]
            
            context_parts.append(f"\n═══ {category.upper().replace('_', ' ')} DATA ═══")

            for i, item in enumerate(items, 1):
                score = item['score']
                relevance = ("🎯 EXACT MATCH" if score > 0.9 else 
                           "🔥 HIGHLY RELEVANT" if score > 0.7 else 
                           "📌 RELEVANT" if score > 0.5 else 
                           "💡 RELATED")
                
                context_parts.append(f"\n{i}. {relevance} (Score: {score:.3f}) | Source: {item['source']}")
                context_parts.append(f"   Content: {item['text']}")
                
                if item['timestamp']:
                    try:
                        dt = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                        context_parts.append(f"   Date: {dt.strftime('%Y-%m-%d')}")
                    except:
                        pass

    total_memories = sum(len(cat_items) for cat_items in categories.values())
    context_parts.insert(0, f"FOUND {total_memories} RELEVANT MEMORIES FROM MICHAEL'S PERSONAL DATABASE:")

    return "\n".join(context_parts) if context_parts else ""

def comprehensive_personal_search(query: str) -> List[Dict]:
    """Comprehensive search for personal/family information across all categories"""
    try:
        # Search multiple variations and categories
        search_variations = [
            query,
            f"family {query}",
            f"personal {query}",
            f"father {query}" if 'father' in query.lower() or 'dad' in query.lower() else None,
            f"mother {query}" if 'mother' in query.lower() or 'mom' in query.lower() else None,
        ]
        
        # Filter out None values
        search_variations = [v for v in search_variations if v]
        
        all_results = []
        
        # Search personal categories specifically
        personal_categories = [
            'personality', 'personality_core', 'profile_core', 'personal_journal',
            'journal_2022', 'journal_2024', 'journal_2025', 'personal_database',
            'family', 'relationships'
        ]
        
        for variation in search_variations:
            # General search
            results = index.query(
                vector=embed(variation),
                top_k=20,
                include_metadata=True
            ).matches
            all_results.extend(results)
            
            # Category-specific searches
            for category in personal_categories:
                try:
                    cat_results = index.query(
                        vector=embed(variation),
                        top_k=10,
                        include_metadata=True,
                        filter={"category": category}
                    ).matches
                    all_results.extend(cat_results)
                except:
                    continue
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            result_id = result.id
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        return unique_results[:25]  # Return top 25 most relevant
        
    except Exception as e:
        logger.error(f"Comprehensive search error: {e}")
        return []

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        logger.info(f"Chat request from {req.session_id}: {req.message[:100]}...")

        # Enhanced memory retrieval with comprehensive search for personal queries
        personal_keywords = ['father', 'dad', 'mother', 'mom', 'family', 'parent', 'brother', 'sister', 
                           'wife', 'husband', 'name', 'born', 'relationship', 'childhood']
        
        is_personal_query = any(keyword.lower() in req.message.lower() for keyword in personal_keywords)
        
        if is_personal_query:
            logger.info("Personal query detected - using comprehensive search")
            memories = comprehensive_personal_search(req.message)
            # Also get regular memories as backup
            regular_memories = fetch_memories(req.message, k=8)
            memories.extend(regular_memories)
        else:
            memories = fetch_memories(req.message, k=12)
            
        memory_context = format_memory_context(memories)

        # Enhanced conversation management
        if req.session_id not in conversation_memory:
            conversation_memory[req.session_id] = []

        system_prompt = """You are Michael's personal AI companion with access to his comprehensive personal database including journals, personality assessments, family information, work documents, and personal history.

ACCURACY RULES - CRITICAL:
🚫 NEVER guess, assume, or fabricate personal information
🚫 NEVER state personal details (names, dates, relationships) unless explicitly found in the provided context
🚫 If you cannot find specific information in the context, you MUST say: "I don't see that specific information in your documented data. Let me search more thoroughly - could you help me understand what you're looking for?"

✅ ONLY state facts that are explicitly written in the RELEVANT CONTEXT section
✅ Always cite the source document when sharing personal information
✅ If context is limited, ask clarifying questions to help retrieve the right information
✅ Prioritize accuracy over appearing knowledgeable

PERSONAL DATA PRIORITY:
- Family information, names, relationships are HIGH PRIORITY for accuracy
- When asked about family members, search thoroughly in personality profiles, journals, and personal documents
- If personal information isn't found, suggest running a more comprehensive search or adding the information to the database

RESPONSE APPROACH:
- Be helpful and conversational
- When uncertain, ask questions to clarify what specific information is needed
- Offer to search different categories or time periods
- Suggest ways to add missing information to the database

Remember: It's better to admit not finding information than to provide incorrect personal details."""

        msgs = [{"role": "system", "content": system_prompt}]

        if memory_context:
            msgs.append({
                "role": "system", 
                "content": f"""RELEVANT CONTEXT FROM MICHAEL'S DOCUMENTED PERSONAL DATA:

{memory_context}

IMPORTANT: You must ONLY use information explicitly stated above. Do not infer, guess, or add details not present in this context. If specific information is not provided above, clearly state that you don't have that information in the documented data."""
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

@app.get("/api/debug/query")
async def debug_query(q: str = "family"):
    """Debug specific queries to see what data is retrieved"""
    try:
        memories = fetch_memories(q, k=15)
        context = format_memory_context(memories)
        
        return {
            "query": q,
            "memories_found": len(memories),
            "memory_scores": [{"score": m.score, "source": m.metadata.get('source', ''), "category": m.metadata.get('category', '')} for m in memories],
            "formatted_context": context,
            "raw_memories": [{"text": m.metadata.get('text', '')[:200], "score": m.score, "source": m.metadata.get('source', '')} for m in memories[:5]]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/personal-data")
async def debug_personal_data(query: str = "family"):
    """Debug endpoint specifically for personal/family data queries"""
    try:
        logger.info(f"Debugging personal data for query: {query}")
        
        # Test comprehensive search
        comprehensive_results = comprehensive_personal_search(query)
        
        # Test regular search
        regular_results = fetch_memories(query, k=15)
        
        # Test category-specific searches
        personal_categories = ['personality', 'personality_core', 'profile_core', 'personal_journal', 
                             'journal_2022', 'journal_2024', 'journal_2025']
        
        category_results = {}
        for category in personal_categories:
            try:
                cat_results = index.query(
                    vector=embed(query),
                    top_k=10,
                    include_metadata=True,
                    filter={"category": category}
                ).matches
                
                category_results[category] = [{
                    "score": r.score,
                    "text_preview": r.metadata.get('text', '')[:200],
                    "source": r.metadata.get('source', ''),
                    "has_family_keywords": any(word in r.metadata.get('text', '').lower() 
                                             for word in ['father', 'dad', 'mother', 'mom', 'parent', 'family'])
                } for r in cat_results]
            except:
                category_results[category] = []
        
        # Extract family mentions from all results
        all_results = comprehensive_results + regular_results
        family_mentions = []
        
        for result in all_results:
            text = result.metadata.get('text', '')
            text_lower = text.lower()
            
            # Look for family-related sentences
            sentences = text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['father', 'dad', 'mother', 'mom', 'parent', 'family']):
                    family_mentions.append({
                        "sentence": sentence.strip(),
                        "source": result.metadata.get('source', ''),
                        "category": result.metadata.get('category', ''),
                        "score": result.score
                    })
        
        return {
            "query": query,
            "comprehensive_search_results": len(comprehensive_results),
            "regular_search_results": len(regular_results),
            "category_breakdown": {cat: len(results) for cat, results in category_results.items()},
            "family_mentions_found": len(family_mentions),
            "family_mentions": family_mentions[:10],  # Top 10 mentions
            "top_comprehensive_results": [{
                "score": r.score,
                "source": r.metadata.get('source', ''),
                "category": r.metadata.get('category', ''),
                "text_preview": r.metadata.get('text', '')[:300]
            } for r in comprehensive_results[:5]],
            "recommendations": [
                "If no family mentions found, check if personal PDFs are processed",
                "Run 'python scripts/validate_personal_data.py' to verify data",
                "Check if MichaelPersonality.pdf and MichaelProfile.pdf are in docs/",
                "Family information should be in personality or profile documents"
            ]
        }
        
    except Exception as e:
        logger.error(f"Personal data debug error: {e}")
        return {"error": str(e)}

@app.get("/api/debug/memory")
async def debug_memory():
    """Debug endpoint to inspect stored memory data"""
    try:
        # Get sample of stored data
        test_embedding = embed("test query")
        results = index.query(
            vector=test_embedding,
            top_k=20,
            include_metadata=True
        )

        debug_data = []
        categories = {}
        sources = {}

        for match in results.matches:
            metadata = match.metadata
            category = metadata.get('category', 'unknown')
            source = metadata.get('source', 'unknown')

            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1

            debug_data.append({
                "score": match.score,
                "source": source,
                "category": category,
                "text_preview": metadata.get('text', 'no text')[:200],
                "timestamp": metadata.get('timestamp', 'unknown'),
                "file_type": metadata.get('file_type', 'unknown')
            })

        # Get index stats
        stats = index.describe_index_stats()

        return {
            "total_vectors": stats.total_vector_count,
            "categories_found": categories,
            "sources_found": sources,
            "sample_data": debug_data,
            "data_quality": {
                "has_text": sum(1 for d in debug_data if d['text_preview'] != 'no text'),
                "has_categories": sum(1 for d in debug_data if d['category'] != 'unknown'),
                "has_sources": sum(1 for d in debug_data if d['source'] != 'unknown')
            }
        }

    except Exception as e:
        logger.error(f"Debug memory error: {e}")
        return {"error": str(e)}

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

@app.post("/api/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text using Whisper"""
    try:
        audio_content = await file.read()
        result = voice_handler.process_voice_message(audio_content)
        return result
    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

@app.post("/api/voice/synthesize")
async def synthesize_speech(request: dict):
    """Convert text to speech"""
    try:
        text = request.get("text", "")
        voice = request.get("voice", "alloy")

        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        audio_content = voice_handler.generate_voice_response(text, voice)

        # Return audio as base64 for web playback
        import base64
        audio_b64 = base64.b64encode(audio_content).decode()

        return {
            "audio_data": audio_b64,
            "content_type": "audio/mpeg"
        }
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing speech: {str(e)}")

@app.post("/api/voice/chat")
async def voice_chat(file: UploadFile = File(...), session_id: str = "default"):
    """Complete voice interaction - transcribe, process, and respond with voice"""
    try:
        # Transcribe audio
        audio_content = await file.read()
        transcription_result = voice_handler.process_voice_message(audio_content)

        if "error" in transcription_result:
            return transcription_result

        transcribed_text = transcription_result["transcribed_text"]

        # Process through chat system
        chat_request = ChatRequest(message=transcribed_text, session_id=session_id)
        chat_response = await chat(chat_request)

        # Generate voice response
        audio_response = voice_handler.generate_voice_response(chat_response["response"])

        import base64
        audio_b64 = base64.b64encode(audio_response).decode()

        return {
            "transcribed_text": transcribed_text,
            "text_response": chat_response["response"],
            "audio_response": audio_b64,
            "content_type": "audio/mpeg"
        }
    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error in voice chat: {str(e)}")

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

@app.get("/api/daily-briefing")
async def get_daily_briefing():
    """Get comprehensive daily briefing with insights and reminders"""
    try:
        briefing = insights_manager.generate_daily_briefing()
        return briefing
    except Exception as e:
        logger.error(f"Daily briefing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating daily briefing: {str(e)}")

@app.post("/api/daily-reflection")
async def add_daily_reflection(reflection: dict):
    """Store daily reflection"""
    try:
        text = reflection.get("text", "")
        mood = reflection.get("mood_score", 5)
        energy = reflection.get("energy_level", 5)

        result = insights_manager.store_daily_reflection(text, mood, energy)
        return result
    except Exception as e:
        logger.error(f"Daily reflection error: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing reflection: {str(e)}")

@app.get("/api/integrations/spotify")
async def get_spotify_data():
    """Get Spotify listening data and insights"""
    try:
        data = integrations.get_spotify_data()
        return data
    except Exception as e:
        logger.error(f"Spotify integration error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting Spotify data: {str(e)}")

@app.get("/api/integrations/hue")
async def get_hue_data():
    """Get Philips Hue lights and environment data"""
    try:
        data = integrations.get_hue_data()
        return data
    except Exception as e:
        logger.error(f"Hue integration error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting Hue data: {str(e)}")

@app.get("/api/integrations/youtube")
async def get_youtube_data():
    """Get YouTube data with OAuth"""
    if 'google_credentials' not in user_sessions:
        return {"error": "Please connect your Google account first", "auth_required": True}

    try:
        data = integrations.get_youtube_data(user_sessions['google_credentials'])
        return {"message": "File processed successfully", "vectors_created": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return {"error": f"Processing failed: {str(e)}"}
    except Exception as e:
        logger.error(f"YouTube integration error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting YouTube data: {str(e)}")

@app.get("/api/integrations/gmail")
async def get_gmail_data():
    """Get Gmail insights"""
    if 'google_credentials' not in user_sessions:
        return {"error": "Please connect your Google account first", "auth_required": True}

    try:
        data = integrations.get_gmail_data(user_sessions['google_credentials'])
        return data
    except Exception as e:
        logger.error(f"Gmail integration error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting Gmail data: {str(e)}")

@app.get("/api/integrations/drive")
async def get_drive_data():
    """Get Google Drive insights"""
    if 'google_credentials' not in user_sessions:
        return {"error": "Please connect your Google account first", "auth_required": True}

    try:
        data = integrations.get_drive_data(user_sessions['google_credentials'])
        return data
    except Exception as e:
        logger.error(f"Drive integration error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting Drive data: {str(e)}")

@app.get("/api/integrations/nas/setup")
async def setup_nas_connection():
    """Setup and test NAS connection"""
    try:
        result = integrations.setup_nas_connection()
        return result
    except Exception as e:
        logger.error(f"NAS setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting up NAS: {str(e)}")

@app.get("/api/integrations/nas/scan")
async def scan_nas_files():
    """Scan NAS for files"""
    try:
        files = integrations.scan_nas_files_advanced()
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"NAS scan error: {e}")
        raise HTTPException(status_code=500, detail=f"Error scanning NAS: {str(e)}")

@app.post("/api/integrations/nas/process")
async def process_nas_file(request: dict):
    """Process a file from NAS and add to memory"""
    try:
        file_path = request.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="file_path is required")

        content = integrations.process_nas_file(file_path)
        if not content:
            return {"error": "Could not extract content from file"}

        # Add to memory system
        from scripts.boot_memory import categorize_content, chunk_text, clean_text

        content = clean_text(content)
        category = categorize_content(content, os.path.basename(file_path))
        chunks = chunk_text(content)

        vector_count = 0
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = embed(chunk)
                if embedding:
                    vector_id = f"nas_{os.path.basename(file_path)}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    metadata = {
                        "text": chunk,
                        "source": f"nas_{os.path.basename(file_path)}",
                        "category": category,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "file_type": "nas_file",
                        "original_path": file_path
                    }
                    index.upsert([(vector_id, embedding, metadata)])
                    vector_count += 1

        return {
            "message": f"Successfully processed {os.path.basename(file_path)}",
            "category": category,
            "chunks_created": vector_count,
            "content_preview": content[:200] + "..." if len(content) > 200 else content
        }

    except Exception as e:
        logger.error(f"NAS file processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing NAS file: {str(e)}")

@app.post("/api/integrations/health/import")
async def import_health_data(file: UploadFile = File(...)):
    """Import Apple Health export data"""
    try:
        # Save uploaded health export
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Parse health data
        health_data = integrations.parse_apple_health_export(temp_path)

        # Clean up temp file
        os.unlink(temp_path)

        if "error" in health_data:
            return health_data

        # Store health insights in memory
        health_summary = f"""Health Data Summary:
Steps: {len(health_data.get('steps', []))} records
Heart Rate: {len(health_data.get('heart_rate', []))} records  
Workouts: {len(health_data.get('workouts', []))} records
Sleep: {len(health_data.get('sleep', []))} records
Weight: {len(health_data.get('weight', []))} records
"""

        vector_id = f"health_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            "text": health_summary,
            "source": "apple_health_import",
            "category": "health",
            "timestamp": datetime.now().isoformat(),
            "file_type": "health_data",
            "data": json.dumps(health_data)
        }

        index.upsert([(vector_id, embed(health_summary), metadata)])

        return {
            "message": "Health data imported successfully",
            "summary": health_data,
            "records_processed": sum(len(health_data.get(key, [])) for key in health_data.keys())
        }

    except Exception as e:
        logger.error(f"Health import error: {e}")
        raise HTTPException(status_code=500, detail=f"Error importing health data: {str(e)}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)