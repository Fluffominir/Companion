# Companion AI — Replit Edition

This repo turns **Michael Slusher’s personal PDFs and notes** into a Retrieval‑Augmented
assistant that you can chat with right inside Replit (or deploy to Fly.io later).

---

## 1 Environment variables

| Variable | Where to get it | Purpose |
|----------|-----------------|---------|
| `OPENAI_API_KEY` | <https://platform.openai.com/account/api-keys> | Embeddings + chat completions + fine‑tune |
| `PINECONE_API_KEY` | <https://app.pinecone.io> → API Keys | Vector database |
| `PINECONE_ENVIRONMENT` | e.g. `gcp-starter` | Cluster region |
| `COMPANION_VOICE_MODEL` | *optional* – set after you fine‑tune | Fine‑tuned model name |

In Replit: **Secrets** → add each key.

---

## 2 Directory layout

```
data/
  raw/               ← drop any new PDF here
  processed/
      knowledge.json
      finetune.jsonl
ingest.py            ← extract + embed + build knowledge.json
build_finetune.py    ← create Q‑A pairs for fine‑tune
main.py              ← FastAPI chat endpoint (sample provided)
requirements.txt
```

---

## 3 One‑time setup

```bash
pip install -r requirements.txt
python ingest.py            # builds data/processed/knowledge.json + Pinecone index
python build_finetune.py    # builds data/processed/finetune.jsonl
# OPTIONAL: fine‑tune
openai file create -p "fine-tune" -f data/processed/finetune.jsonl
openai fine_tuning.jobs.create -t <FILE_ID> -m gpt-3.5-turbo-0125
```

---

## 4 Running locally in the Repl

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the **WebView** tab → you get `/docs` (Swagger UI).

---

## 5 Nightly refresh (optional)

Add this to Replit’s **Nix cron** or create a `@replit/background` repl:

```bash
python ingest.py
```

Any new PDFs dropped into `data/raw/` will be indexed automatically.

---

## 6 Notes

* **Handwriting OCR** – `ingest.py` will OCR pages with <40 characters of raw text.
* **Source traceability** – every chunk carries `source_file#page` so citations remain intact.
* **Security** – Never commit your raw medical PDFs; keep them in your Repl’s private storage.
