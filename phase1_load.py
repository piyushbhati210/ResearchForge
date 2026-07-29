# ============================================================
# PHASE 1 — Load and Clean PDFs
# ============================================================

# WHY pdfplumber?
# pdfplumber is the best free library for extracting text from PDFs.
# It handles multi-column layouts, tables, and special characters
# much better than PyPDF2 or pdfminer.
import pdfplumber

# WHY os?
# os lets us work with files and folders on your computer.
# We use it to loop through all PDFs in the papers/ folder.
import os

# WHY re?
# re is Python's regular expressions library.
# We use it to clean up messy text — remove extra spaces,
# weird characters, and formatting problems from PDFs.
import re

# WHY json?
# We save our cleaned chunks as JSON so we can load them
# in Phase 2 without re-processing all PDFs again.
import json


# ── FUNCTION 1: Load one PDF ──────────────────────────────
def load_pdf(filepath):
    """
    Opens one PDF file and extracts all text from all pages.
    
    WHY page by page?
    Some PDFs have 50+ pages. Reading page by page prevents
    memory errors and lets us track which page text comes from.
    """
    text = ""
    
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_number, page in enumerate(pdf.pages):
                
                # extract_text() converts the PDF page to a string
                # We use "or empty string" because some pages have
                # only images and return None — this prevents errors
                page_text = page.extract_text() or ""
                
                # Add page separator so we know where pages end
                text += f"\n--- Page {page_number + 1} ---\n"
                text += page_text
                
    except Exception as e:
        # If a PDF is corrupted or password-protected, skip it
        # and print the error instead of crashing everything
        print(f"ERROR loading {filepath}: {e}")
        return ""
    
    return text


# ── FUNCTION 2: Clean the extracted text ─────────────────
def clean_text(text):
    """
    Removes noise from extracted PDF text.
    
    WHY clean?
    Raw PDF text has lots of problems:
    - Multiple spaces between words
    - Newlines in the middle of sentences
    - Headers/footers repeated on every page
    - Special unicode characters that break the LLM
    Cleaning makes the text much better for the AI to understand.
    """
    
    # Remove multiple spaces → replace with single space
    # WHY: PDFs often have "word     word" which confuses embeddings
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines → replace with double newline
    # WHY: Keeps paragraph breaks but removes excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove special unicode characters that are not normal text
    # WHY: Things like ∂, ∑, © cause errors in some tokenisers
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u2018-\u201D]', ' ', text)
    
    # Remove very short lines (page numbers, headers like "2 | Page")
    # WHY: These add noise without adding any meaning
    lines = text.split('\n')
    lines = [line for line in lines if len(line.strip()) > 15]
    text = '\n'.join(lines)
    
    # Final strip to remove leading/trailing whitespace
    text = text.strip()
    
    return text


# ── FUNCTION 3: Split text into chunks ───────────────────
def split_into_chunks(text, chunk_size=500, overlap=50):
    """
    Splits long text into smaller overlapping chunks.
    
    WHY chunk?
    LLMs have a context window limit — they cannot read 50 pages
    at once. We split text into small pieces so the LLM only reads
    the relevant piece when answering a question.
    
    WHY overlap=50?
    If a sentence is split across two chunks, the overlap ensures
    both chunks have enough context to understand that sentence.
    Without overlap, we would lose meaning at chunk boundaries.
    
    WHY chunk_size=500?
    500 words is:
    - Small enough to fit in LLM context
    - Large enough to contain a complete idea
    - Standard size used in most RAG systems
    """
    
    # Split by words
    words = text.split()
    chunks = []
    
    # Slide a window of chunk_size words across the text
    # Step size = chunk_size - overlap (so chunks overlap by 50 words)
    for i in range(0, len(words), chunk_size - overlap):
        
        # Take chunk_size words starting at position i
        chunk_words = words[i:i + chunk_size]
        
        # Join words back into a string
        chunk_text = ' '.join(chunk_words)
        
        # Only keep chunks that have meaningful content
        # WHY 100 words minimum? Chunks smaller than this are usually
        # just page headers or incomplete sentences — not useful
        if len(chunk_words) >= 100:
            chunks.append(chunk_text)
    
    return chunks


# ── FUNCTION 4: Process all PDFs ─────────────────────────
def process_all_pdfs(papers_folder="papers/"):
    """
    Loops through all PDFs in the papers folder,
    loads, cleans, and chunks each one.
    
    Returns a list of dictionaries, each containing:
    - text: the chunk content
    - source: which PDF it came from
    - chunk_id: unique identifier
    """
    
    all_chunks = []
    chunk_counter = 0
    
    # Get list of all PDF files in the folder
    pdf_files = [f for f in os.listdir(papers_folder) 
                 if f.endswith('.pdf')]
    
    print(f"\nFound {len(pdf_files)} PDF files")
    print("=" * 50)
    
    for pdf_file in pdf_files:
        filepath = os.path.join(papers_folder, pdf_file)
        
        print(f"\nProcessing: {pdf_file}")
        
        # Step 1: Load the PDF
        raw_text = load_pdf(filepath)
        
        if not raw_text:
            print(f"  Skipping — could not read file")
            continue
        
        print(f"  Extracted: {len(raw_text)} characters")
        
        # Step 2: Clean the text
        clean = clean_text(raw_text)
        print(f"  After cleaning: {len(clean)} characters")
        
        # Step 3: Split into chunks
        chunks = split_into_chunks(clean, chunk_size=500, overlap=50)
        print(f"  Created: {len(chunks)} chunks")
        
        # Step 4: Save each chunk with metadata
        for chunk in chunks:
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "source": pdf_file,           # which paper
                "text": chunk                  # the actual content
            })
            chunk_counter += 1
    
    print(f"\n{'=' * 50}")
    print(f"TOTAL: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    
    return all_chunks


# ── MAIN: Run Phase 1 ─────────────────────────────────────
if __name__ == "__main__":
    
    # Process all PDFs
    chunks = process_all_pdfs(papers_folder="papers/")
    
    # Save chunks to JSON file
    # WHY save to JSON?
    # So we do not re-process PDFs every time we run Phase 2.
    # JSON is human-readable so you can check what was extracted.
    with open("outputs/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to outputs/chunks.json")
    print(f"Phase 1 Complete!")
    
    # Show a sample chunk so you can verify it looks correct
    if chunks:
        print(f"\n--- SAMPLE CHUNK ---")
        print(f"Source: {chunks[0]['source']}")
        print(f"Text preview: {chunks[0]['text'][:300]}...")