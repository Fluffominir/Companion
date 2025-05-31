"""
ingest.py — extracts all PDFs in data/raw, builds knowledge.json and
upserts embeddings to Pinecone ("companion-memory" index, namespace "v1").
"""

import os, json, re, uuid, io, datetime as dt, pathlib
from pypdf import PdfReader
from PIL import Image
import pytesseract, openai, dotenv
import pinecone

dotenv.load_dotenv()
client = openai.OpenAI()
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("companion-memory")

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
KN_PATH = OUT_DIR/"knowledge.json"

knowledge = {"meta":{"version":dt.date.today().isoformat(),
                     "owner":"Michael Slusher"},"documents":[]}

def page_text(page):
    text = page.extract_text() or ""
    if len(text.strip()) < 40 and page.images:
        img = Image.open(io.BytesIO(page.images[0].data))
        text = pytesseract.image_to_string(img)
    return text

for pdf in RAW_DIR.glob("*.pdf"):
    reader = PdfReader(str(pdf))
    doc_id = pdf.stem.replace(" ", "_").lower()
    doc = {"doc_id":doc_id,"title":pdf.stem,
           "source_file":pdf.name,"pages":[]}
    for i, page in enumerate(reader.pages, start=1):
        text = page_text(page)
        for para in re.split(r"\n{2,}", text):
            if len(para.strip()) < 10:
                continue
            chunk_id = f"{doc_id}_p{i}_{uuid.uuid4().hex[:6]}"
            doc["pages"].append({
                "page": i,
                "chunks": [{
                    "chunk_id": chunk_id,
                    "type": "journal" if para.startswith(("I ","Today","We ")) else "fact",
                    "label": para.split(":")[0][:40],
                    "value": para.strip(),
                    "source": f"{pdf.name}#p{i}"
                }]
            })
            # embed + upsert
            emb = client.embeddings.create(
                model="text-embedding-3-small", input=para
            ).data[0].embedding
            index.upsert([{ "id": chunk_id,
                            "values": emb,
                            "metadata": {"text": para,
                                         "source": f"{pdf.name}#p{i}"} }],
                         namespace="v1")
    knowledge["documents"].append(doc)

json.dump(knowledge, open(KN_PATH,"w"), indent=2)
print("✓ knowledge.json built and embeddings stored")
