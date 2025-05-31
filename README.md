
# Michael's AI Companion

A personal AI assistant trained on your personal data using RAG (Retrieval-Augmented Generation).

## Quick Setup

1. **Add your personal data:**
   - Place PDFs, markdown files, or text files in `docs/` or `attached_assets/`
   - Fill out `docs/core_personal_data.md` with your basic information

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export PINECONE_API_KEY=your_pinecone_key
   ```

3. **Process your data:**
   ```bash
   python scripts/clean_ingest.py
   ```

4. **Start the companion:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open your companion:**
   Go to `http://localhost:8000`

## Commands

- **Ingest new data:** `python scripts/clean_ingest.py`
- **Validate system:** `python scripts/validate_data.py`
- **Test specific query:** `python scripts/validate_data.py "your question here"`

## Features

- ✅ Clean RAG-based retrieval
- ✅ Accuracy-focused responses (no hallucination)
- ✅ Support for PDFs, markdown, and text files
- ✅ Automatic categorization
- ✅ Debug endpoints for troubleshooting

## File Structure

```
docs/                       # Your personal documents (PDFs, markdown, text)
├── core_personal_data.md   # Fill this out first!
└── [your personal files]   # Journals, books, profiles, etc.

attached_assets/            # Additional personal documents
├── MichaelPersonality.pdf  # Core personality data
└── MichaelProfile.pdf      # Core profile data

scripts/
├── clean_ingest.py         # Process and index documents
└── validate_data.py        # Test and validate system

main.py                     # FastAPI application
static/index.html          # Web interface
.env.template              # Environment variables template
requirements.txt           # Python dependencies
```

## Adding New Data

Simply add files to `docs/` or `attached_assets/` and run:
```bash
python scripts/clean_ingest.py
```

The system will automatically:
- Extract text from PDFs, markdown, and text files
- Categorize content (family, work, goals, etc.)
- Create searchable embeddings
- Make it available to your AI companion

## Troubleshooting

If your AI can't find information:
1. Run `python scripts/validate_data.py` to check data
2. Ensure your files contain readable text (not just images)
3. Re-run `python scripts/clean_ingest.py` to reprocess
4. Test specific queries: `python scripts/validate_data.py "family information"`
