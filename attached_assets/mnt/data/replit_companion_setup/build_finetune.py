"""
build_finetune.py — generates Q‑A pairs from knowledge.json
"""

import json, jsonlines, pathlib
KN_PATH = pathlib.Path("data/processed/knowledge.json")
OUT_PATH = pathlib.Path("data/processed/finetune.jsonl")
src = json.load(open(KN_PATH))
with jsonlines.open(OUT_PATH, "w") as out:
    for doc in src["documents"]:
        for page in doc["pages"]:
            for c in page["chunks"]:
                if c["type"] != "fact":
                    continue
                q = f"What is Michael's {c['label'].lower()}?"
                a = c["value"].strip()
                out.write({"messages":[
                    {"role":"system","content":"You are Michael’s factual assistant—for personal use only."},
                    {"role":"user","content":q},
                    {"role":"assistant","content":a}
                ]})
print("✓ finetune.jsonl written")
