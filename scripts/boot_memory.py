import glob, os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(INDEX_NAME)

def embed(text):  
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

for path in glob.glob("docs/**/*.md", recursive=True):
    with open(path) as f:
        txt = f.read()
    index.upsert([(path, embed(txt), {"text": txt})])

print("Synced", index.describe_index_stats().total_vector_count, "vectors")
