import pdfplumber
from pythainlp.util import normalize
from pythainlp import word_tokenize
import io
import re
import streamlit as st
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_content):
    """Extract text from PDF with enhanced Thai language support."""
    try:
        pdf_file = io.BytesIO(file_content)
        text = []

        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = normalize_thai_text(page_text)
                        text.append(page_text)
                except Exception as e:
                    st.warning(f"Warning: Could not extract text from a page: {str(e)}")
                    continue

        if not text:
            pdf_file.seek(0)
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = normalize_thai_text(page_text)
                        text.append(page_text)
                except Exception as e:
                    st.warning(
                        f"Warning: Fallback extraction failed for a page: {str(e)}"
                    )
                    continue

        full_text = "\n".join(text)
        full_text = normalize_thai_text(full_text)

        return full_text if full_text else None
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None


def normalize_thai_text(text):
    """Normalize Thai text by cleaning and standardizing characters."""
    text = normalize(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s.]", "", text)
    return text.strip()


# เพิ่มฟังก์ชั่น chunk_thai_text ที่ขาดหายไป
def chunk_thai_text(text, chunk_size=800, overlap=80):
    """ปรับปรุงการแบ่งข้อความภาษาไทยเป็นส่วนๆ"""
    # Normalize text
    text = normalize_thai_text(text)

    # Split text into sentences first
    sentences = text.split(".")

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Clean the sentence
        sentence = sentence.strip()
        if not sentence:
            continue

        # Tokenize sentence
        words = word_tokenize(sentence)
        sentence_length = len(words)

        # If adding this sentence exceeds chunk size
        if current_length + sentence_length > chunk_size:
            # Save current chunk if it's not empty
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

            # Start new chunk with overlap
            if chunks and overlap > 0:
                # Get last few words from previous chunk for overlap
                last_chunk_words = word_tokenize(chunks[-1])
                overlap_words = last_chunk_words[-overlap:]
                current_chunk = overlap_words + words
            else:
                current_chunk = words

            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += sentence_length

    # Add the last chunk if not empty
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks
