
import glob, os
import pypdf
import pdfplumber
from PIL import Image
import easyocr
import pytesseract
from docx import Document
import openpyxl
from pptx import Presentation
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import re
from datetime import datetime
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "companion-memory"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pc.Index(INDEX_NAME)

# Initialize OCR reader
try:
    ocr_reader = easyocr.Reader(['en'])
    print("EasyOCR initialized for handwriting recognition")
except Exception as e:
    print(f"OCR initialization warning: {e}")
    ocr_reader = None

def embed(text):  
    try:
        return client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def categorize_content(text, filename):
    """Enhanced categorization with better logic"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Priority-based categorization
    if any(word in filename_lower for word in ['journal', 'diary']):
        return 'personal_journal'
    elif any(word in filename_lower for word in ['personality', 'profile', 'michael']):
        return 'personality'
    elif any(word in filename_lower for word in ['health', 'medical', 'superbill']):
        return 'health'
    elif any(word in filename_lower for word in ['company', 'employee', 'handbook', 'rocket', 'launch']):
        return 'work'
    elif any(word in filename_lower for word in ['pitch', 'serwm', 'brand']):
        return 'projects'
    elif any(word in filename_lower for word in ['attached', 'body', 'myth', 'steal', 'show', 'emyth']):
        return 'books'
    elif any(word in text_lower for word in ['goal', 'objective', 'want to', 'plan to', 'achieve']):
        return 'goals'
    elif any(word in text_lower for word in ['meeting', 'call', 'appointment', 'schedule']):
        return 'meetings'
    elif any(word in text_lower for word in ['project', 'task', 'todo', 'work on']):
        return 'projects'
    elif any(word in text_lower for word in ['idea', 'thought', 'concept', 'brainstorm']):
        return 'ideas'
    else:
        return 'general'

def extract_text_with_ocr(file_path):
    """Extract text from images using OCR for handwriting recognition"""
    try:
        if ocr_reader:
            # Use EasyOCR for better handwriting recognition
            results = ocr_reader.readtext(file_path)
            text = ' '.join([result[1] for result in results])
            return text
        else:
            # Fallback to pytesseract
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
    except Exception as e:
        print(f"OCR error for {file_path}: {e}")
        return ""

def extract_pdf_text_advanced(pdf_path):
    """Advanced PDF text extraction with multiple methods"""
    text = ""
    
    # Method 1: Try pdfplumber (better for complex layouts)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDFPlumber failed for {pdf_path}: {e}")
    
    # Method 2: Fallback to pypdf if pdfplumber didn't work well
    if len(text.strip()) < 100:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                fallback_text = ""
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            fallback_text += page_text + "\n"
                    except Exception as e:
                        continue
                if len(fallback_text) > len(text):
                    text = fallback_text
        except Exception as e:
            print(f"PyPDF fallback failed for {pdf_path}: {e}")
    
    return text.strip()

def extract_docx_text(docx_path):
    """Extract text from Word documents"""
    try:
        doc = Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_xlsx_text(xlsx_path):
    """Extract text from Excel files"""
    try:
        workbook = openpyxl.load_workbook(xlsx_path)
        text = []
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            text.append(f"Sheet: {sheet_name}")
            for row in sheet.iter_rows(values_only=True):
                row_text = [str(cell) if cell is not None else "" for cell in row]
                text.append(" | ".join(row_text))
        return '\n'.join(text)
    except Exception as e:
        print(f"Error reading XLSX {xlsx_path}: {e}")
        return ""

def extract_pptx_text(pptx_path):
    """Extract text from PowerPoint files"""
    try:
        prs = Presentation(pptx_path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return '\n'.join(text)
    except Exception as e:
        print(f"Error reading PPTX {pptx_path}: {e}")
        return ""

def chunk_text(text, max_length=600):
    """Improved text chunking with better sentence boundaries"""
    if len(text) <= max_length:
        return [text]
    
    # Try to split on sentence boundaries first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 2 <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]

def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

vector_count = 0
processed_files = []
categories_used = set()

print("Starting comprehensive memory synchronization...")
print(f"Working directory: {os.getcwd()}")
print(f"Looking for files in: {os.path.abspath('docs')}")

# Process markdown files
md_files = glob.glob("docs/**/*.md", recursive=True)
print(f"Found {len(md_files)} markdown files: {md_files}")

for path in md_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        
        if not txt.strip():
            continue
            
        txt = clean_text(txt)
        category = categorize_content(txt, path)
        categories_used.add(category)
        chunks = chunk_text(txt)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            embedding = embed(chunk)
            if embedding is None:
                continue
                
            vector_id = f"{path}_{i}" if len(chunks) > 1 else path.replace('/', '_').replace(' ', '_')
            metadata = {
                "text": chunk,
                "source": path,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "file_type": "markdown"
            }
            
            index.upsert([(vector_id, embedding, metadata)])
            vector_count += 1
        
        processed_files.append(path)
        print(f"✓ Processed: {path} ({category})")
        
    except Exception as e:
        print(f"✗ Error processing {path}: {e}")

# Process PDF files
pdf_files = glob.glob("docs/**/*.pdf", recursive=True)
print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

for path in pdf_files:
    try:
        print(f"Processing PDF: {path}")
        txt = extract_pdf_text_advanced(path)
        
        if not txt.strip():
            print(f"No text extracted from {path}")
            continue
            
        txt = clean_text(txt)
        category = categorize_content(txt, path)
        categories_used.add(category)
        chunks = chunk_text(txt, max_length=800)
        
        processed_chunks = 0
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            embedding = embed(chunk)
            if embedding is None:
                continue
                
            vector_id = f"{path}_{i}".replace('/', '_').replace(' ', '_')
            metadata = {
                "text": chunk,
                "source": path,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "file_type": "pdf"
            }
            
            index.upsert([(vector_id, embedding, metadata)])
            vector_count += 1
            processed_chunks += 1
        
        processed_files.append(path)
        print(f"✓ Processed: {path} ({category}) - {processed_chunks} chunks")
        
    except Exception as e:
        print(f"✗ Error processing PDF {path}: {e}")

# Process Office documents
office_extensions = [
    ('docs/**/*.docx', extract_docx_text),
    ('docs/**/*.xlsx', extract_xlsx_text),
    ('docs/**/*.pptx', extract_pptx_text)
]

for pattern, extract_func in office_extensions:
    files = glob.glob(pattern, recursive=True)
    for path in files:
        try:
            print(f"Processing Office document: {path}")
            txt = extract_func(path)
            
            if not txt.strip():
                continue
                
            txt = clean_text(txt)
            category = categorize_content(txt, path)
            categories_used.add(category)
            chunks = chunk_text(txt)
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                embedding = embed(chunk)
                if embedding is None:
                    continue
                    
                vector_id = f"{path}_{i}".replace('/', '_').replace(' ', '_')
                metadata = {
                    "text": chunk,
                    "source": path,
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat(),
                    "file_type": path.split('.')[-1]
                }
                
                index.upsert([(vector_id, embedding, metadata)])
                vector_count += 1
            
            processed_files.append(path)
            print(f"✓ Processed: {path} ({category})")
            
        except Exception as e:
            print(f"✗ Error processing {path}: {e}")

# Process images for handwriting recognition
image_files = glob.glob("docs/**/*.{jpg,jpeg,png,tiff,bmp}", recursive=True)
if image_files:
    print(f"Found {len(image_files)} image files for OCR processing")
    
    for path in image_files:
        try:
            print(f"Processing image with OCR: {path}")
            txt = extract_text_with_ocr(path)
            
            if not txt.strip() or len(txt) < 10:
                continue
                
            txt = clean_text(txt)
            category = categorize_content(txt, path)
            categories_used.add(category)
            
            embedding = embed(txt)
            if embedding is None:
                continue
                
            vector_id = path.replace('/', '_').replace(' ', '_')
            metadata = {
                "text": txt,
                "source": path,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "file_type": "image_ocr"
            }
            
            index.upsert([(vector_id, embedding, metadata)])
            vector_count += 1
            processed_files.append(path)
            print(f"✓ Processed OCR: {path} ({category})")
            
        except Exception as e:
            print(f"✗ Error processing image {path}: {e}")

total_files = len(processed_files)
print(f"\n🎉 Memory sync complete!")
print(f"📁 Successfully processed {total_files} files")
print(f"🧠 Created {vector_count} memory vectors")
print(f"📂 Categories used: {sorted(categories_used)}")
print(f"📊 Average vectors per file: {vector_count/total_files if total_files > 0 else 0:.1f}")
