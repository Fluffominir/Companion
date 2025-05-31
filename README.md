
# Personal AI Companion

A streamlined personal AI assistant using RAG (Retrieval-Augmented Generation) trained on your personal documents.

## Quick Start

1. **Setup environment:**
   ```bash
   export OPENAI_API_KEY=your_key
   export PINECONE_API_KEY=your_key
   ```

2. **Add your data:**
   - Fill out `docs/core_personal_data.md`
   - Add PDFs/documents to `docs/` or `attached_assets/`

3. **Process data:**
   ```bash
   python scripts/clean_ingest.py
   ```

4. **Run application:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Commands

| Command | Purpose |
|---------|---------|
| `python scripts/clean_ingest.py` | Process new documents |
| `python scripts/validate_data.py` | Test system health |
| `python scripts/validate_data.py "query"` | Test specific search |

## Features

- 📄 Supports PDFs, Markdown, Text files
- 🎯 Accurate retrieval without hallucination
- 🔍 Automatic content categorization
- 🔧 Built-in validation and testingleshooting

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
