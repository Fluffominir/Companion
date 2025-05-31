"""
ingest.py
• Extracts every PDF in data/raw/
• OCRs pages that have little or no embedded text
• Chunks ~400 tokens with 200-token overlap
• Stores {id, embedding, metadata} in Pinecone ("companion-memory", namespace "v1")
• Saves a local knowledge.json for debugging
"""
import os, io, re, json, uuid, pathlib, datetime as dt
from typing import List
from pypdf import PdfReader
from PIL import Image
import pytesseract, openai, pinecone
from dotenv import load_dotenv

# ------------ CONFIG ------------
load_dotenv()
RAW_DIR   = pathlib.Path("data/raw")
OUT_DIR   = pathlib.Path("data/processed"); OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_NAME = "companion-memory"
NAMESPACE  = "v1"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE  = 400   # tokens ~≈ words here
OVERLAP     = 200
# ---------------------------------

openai_client = openai.OpenAI()
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
try:
    index = pc.Index(INDEX_NAME)
except pinecone.exceptions.NotFoundException:
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine")
    index = pc.Index(INDEX_NAME)

def page_text(page) -> str:
    txt = page.extract_text() or ""
    if len(txt.strip()) < 40 and page.images:
        img = Image.open(io.BytesIO(page.images[0].data))
        txt = pytesseract.image_to_string(img)
    return txt.replace("\r", "\n")

def chunk_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], []
    count = 0
    for p in paras:
        words = p.split()
        if count + len(words) > CHUNK_SIZE:
            chunks.append(" ".join(buf))
            buf, count = buf[-OVERLAP//2:], sum(len(w) for w in buf[-OVERLAP//2:])
        buf.extend(words)
        count += len(words)
    if buf:
        chunks.append(" ".join(buf))
    return chunks

knowledge = {"meta":{"version":dt.date.today().isoformat(),
                     "owner":"Michael Slusher"},
             "documents":[]}

for pdf in RAW_DIR.glob("*.pdf"):
    reader = PdfReader(str(pdf))
    doc_id = pdf.stem.replace(" ", "_").lower()
    doc = {"doc_id":doc_id,"title":pdf.stem,
           "source_file":pdf.name,"pages":[]}
    for i, page in enumerate(reader.pages, start=1):
        text = page_text(page)
        for chunk in chunk_paragraphs(text):
            chunk_id = f"{doc_id}_p{i}_{uuid.uuid4().hex[:6]}"
            meta = {"text": chunk[:300], "source": f"{pdf.name}#p{i}"}
            # embed & upsert
            emb = openai_client.embeddings.create(
                model=EMBED_MODEL, input=chunk
            ).data[0].embedding
            index.upsert([{"id":chunk_id,"values":emb,"metadata":meta}],
                         namespace=NAMESPACE)
            doc["pages"].append({"page":i,"chunk_id":chunk_id,"label":meta["text"]})
    knowledge["documents"].append(doc)

json.dump(knowledge, open(OUT_DIR/"knowledge.json","w"), indent=2)
print("✓ ingest complete → vectors in Pinecone, knowledge.json written.")
