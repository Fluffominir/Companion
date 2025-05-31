
"""
build_finetune.py — Generate fine-tuning data from your personal documents
"""

import json
import jsonlines
import pathlib
import os
from pypdf import PdfReader
import re

def extract_facts_from_text(text, source_file):
    """Extract facts and personal information from text"""
    facts = []
    
    # Clean text
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Split into paragraphs
    paragraphs = re.split(r'\n{2,}', text)
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50 or len(para) > 2000:  # Skip very short or very long paragraphs
            continue
            
        # Look for fact-like statements
        if any(keyword in para.lower() for keyword in [
            'i am', 'my name', 'i work', 'i live', 'my goal', 
            'i like', 'i prefer', 'my hobby', 'my interest',
            'my family', 'my wife', 'my children', 'my job'
        ]):
            facts.append({
                "text": para[:1500],  # Limit length
                "source": source_file,
                "type": "personal_fact"
            })
        elif para.startswith(('I ', 'Today ', 'We ', 'My ', 'This ')):
            facts.append({
                "text": para[:1500],  # Limit length
                "source": source_file,
                "type": "journal_entry"
            })
    
    return facts

def main():
    print("🔨 Building fine-tuning dataset...")
    
    # Collect all facts
    all_facts = []
    
    # Process PDFs from both directories
    source_dirs = ["docs/", "attached_assets/"]
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        for pdf_file in pathlib.Path(source_dir).glob("*.pdf"):
            print(f"📄 Processing {pdf_file.name}")
            
            try:
                reader = PdfReader(str(pdf_file))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                if text.strip():
                    facts = extract_facts_from_text(text, pdf_file.name)
                    all_facts.extend(facts)
                    print(f"  ✅ Extracted {len(facts)} facts")
                    
            except Exception as e:
                print(f"  ❌ Error processing {pdf_file.name}: {e}")
    
    # Create fine-tuning JSONL
    output_file = "finetune_data.jsonl"
    
    with jsonlines.open(output_file, mode='w') as writer:
        for fact in all_facts:
            # Create Q&A pairs
            if fact["type"] == "personal_fact":
                question = f"Tell me about Michael's personal information."
                answer = fact["text"]
            elif fact["type"] == "journal_entry":
                question = f"What did Michael write in his journal?"
                answer = fact["text"]
            else:
                continue
                
            # Write training example
            writer.write({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Michael's personal AI companion with access to his private information. Respond naturally and personally."
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            })
    
    print(f"\n🎉 Created {output_file} with {len(all_facts)} training examples")
    print("\nTo create a fine-tuned model:")
    print("1. Upload the file: openai files create finetune_data.jsonl")
    print("2. Start fine-tuning: openai fine_tuning.jobs.create -t <file_id> -m gpt-3.5-turbo")

if __name__ == "__main__":
    main()
