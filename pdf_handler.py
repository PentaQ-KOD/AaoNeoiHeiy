import pdfplumber
from pythainlp.util import normalize
from pythainlp.tokenize import sent_tokenize, word_tokenize
import io
import re
import streamlit as st
import fitz  # PyMuPDF
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_content):
    """Extract text from PDF with enhanced Thai language support."""
    pdf_file = io.BytesIO(file_content)
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    text = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        if page_text:
            text.append(page_text)
        else:
            st.warning(f"⚠️ PyMuPDF อ่านหน้าที่ {i+1} ไม่ได้ (อาจเป็นภาพ)")

    return "\n".join(text) if text else None


def normalize_thai_text(text):
    """Normalize Thai text by cleaning and standardizing characters."""
    text = normalize(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s.,!?()]", "", text)
    return text.strip()


# เพิ่มฟังก์ชั่น chunk_thai_text ที่ขาดหายไป
def chunk_thai_text(text, chunk_size=800, overlap=80):
    """ปรับปรุงการแบ่งข้อความภาษาไทยเป็นส่วนๆ"""
    # Normalize text
    text = normalize_thai_text(text)

    # Split text into sentences first
    sentences = sent_tokenize(
        text, engine="whitespace"
    )  # ใช้ whitespace engine เพื่อรักษาความหมาย

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Clean the sentence
        sentence = sentence.strip()
        if not sentence:
            continue

        # Tokenize sentence
        words = word_tokenize(sentence, engine="newmm")
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
