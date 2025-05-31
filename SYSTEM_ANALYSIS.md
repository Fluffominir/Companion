
# AI Companion System - Technical Analysis

## For ChatGPT Analysis

This document provides a comprehensive technical overview of Michael Slusher's personal AI companion system.

## Core Architecture

### 1. RAG Pipeline (`main.py`)
```python
# Main components:
- FastAPI web application
- OpenAI GPT-4o for responses  
- OpenAI text-embedding-3-small for search
- Pinecone vector database (693 vectors)
- Conversation memory management
- Source attribution system
```

### 2. Data Processing (`scripts/clean_ingest.py`)
```python
# Document processing workflow:
1. Scan docs/ and attached_assets/ directories
2. Extract text from PDFs, markdown, text files
3. Clean and normalize text content
4. Split into 500-word chunks with 50-word overlap
5. Generate embeddings using OpenAI
6. Store in Pinecone with metadata categorization
7. Currently: 693 vectors from 25+ source files
```

### 3. Knowledge Categories
```python
categories = {
    "personality": "MBTI results, core traits, assessments",
    "journal": "Personal entries from 2022-2025",
    "work": "Rocket Launch Studio employment info", 
    "goals": "Personal objectives and planning",
    "health": "Medical records and wellness data",
    "profile": "Basic biographical information",
    "family": "Family relationships and dynamics"
}
```

## System Prompts & Accuracy Focus

### Primary System Prompt
```
You are Michael's personal AI companion with access to his comprehensive personal database.

ACCURACY RULES - CRITICAL:
🚫 NEVER guess, assume, or fabricate personal information
🚫 NEVER state personal details unless explicitly found in the provided context
🚫 If you cannot find specific information, say: "I don't see that information in your documented data"

✅ ONLY state facts that are explicitly written in the context
✅ Always cite the source when sharing personal information  
✅ When uncertain, ask clarifying questions
✅ Be helpful and conversational while maintaining accuracy
```

## Current Performance

### Data Validation Results
```bash
# Latest validation (successful):
✅ Database health: 693 vectors indexed
✅ Search functionality: Working correctly
✅ Response accuracy: 5/5 test queries passed
✅ Source attribution: All responses properly cited
✅ No hallucinations detected in testing
```

### Technical Metrics
- **Response latency**: 2-3 seconds average
- **Vector search**: 8 results retrieved, filtered to top relevant
- **Context window**: ~8000 characters max per response
- **Conversation memory**: 20 messages rolling buffer
- **Similarity threshold**: 0.3 minimum for relevance

## Key Technical Decisions

### 1. Accuracy Over Creativity
- Strict prompt engineering to prevent hallucination
- Source citation requirements for all personal facts
- Conservative similarity thresholds
- Explicit "I don't know" responses when data unavailable

### 2. Personal Data Structure
```python
# Metadata structure for each vector:
{
    "text": "The actual content chunk",
    "source": "Full file path",
    "category": "Auto-detected content type", 
    "chunk_id": "Unique identifier",
    "created_at": "Processing timestamp"
}
```

### 3. Retrieval Strategy
```python
# Multi-step retrieval process:
1. Generate query embedding
2. Search Pinecone with top_k=16 
3. Filter results by similarity > 0.3
4. Deduplicate similar content
5. Format top 8 results for context
6. Inject into GPT-4o conversation
```

## Web Interface (`static/index.html`)

### Features Implemented
- Clean chat interface with message history
- Real-time typing indicators  
- Mobile-responsive design
- Conversation export functionality
- System status monitoring
- Error handling and retry logic

### Frontend Architecture
```javascript
// Key components:
- Message handling with fetch API
- Dynamic DOM updates for chat flow
- Local conversation storage
- Status polling for system health
- Export functionality for chat history
```

## Development Workflow

### Adding New Data
```bash
1. Add files to docs/ or attached_assets/
2. Run: python scripts/clean_ingest.py
3. Validate: python scripts/validate_data.py
4. Test queries through web interface
```

### System Monitoring
```bash
# Health checks available:
GET /api/status          # System overview
GET /api/debug/search    # Test search function
GET /health              # Basic uptime check
```

## Known Limitations

1. **Vector Storage**: Limited to 693 vectors currently
2. **File Types**: Only PDF, Markdown, Text supported
3. **Context Window**: 8000 character limit per response
4. **Memory**: Conversation limited to 20 messages
5. **Real-time**: No live data integration yet

## Future Enhancement Opportunities

### Immediate (Week 1-2)
- Add more personal documents
- Implement conversation threading
- Enhanced error handling
- Performance optimizations

### Medium-term (Month 1-3)  
- Voice interface integration
- Mobile app development
- Advanced analytics dashboard
- Multi-modal support (images)

### Long-term (3+ months)
- Real-time data integration (Calendar, Email)
- Predictive insights
- Automated data ingestion
- Fine-tuned personal model

## Security & Privacy

### Current Implementation
- Environment variables for API keys
- No personal data in public repositories  
- Local conversation storage
- Pinecone data encryption at rest

### Access Control
- Single-user system (Michael)
- No authentication layer currently
- Private Replit deployment
- GitHub repository controls

## Testing & Validation

### Automated Tests
```python
# test_queries implemented:
1. "What are my core personality traits?"
2. "Tell me about my family relationships"  
3. "What are my career goals?"
4. "What health issues should I monitor?"
5. "What did I accomplish in 2024?"
```

### Manual Testing Process
1. Query system with known personal facts
2. Verify source attribution accuracy
3. Check for hallucinated information
4. Validate conversation memory
5. Test edge cases and error handling

---

This system represents a sophisticated personal AI implementation focused on accuracy, privacy, and usefulness for daily personal assistance tasks.
