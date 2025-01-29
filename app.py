import streamlit as st
import google.generativeai as genai
import os
import tempfile
from PyPDF2 import PdfReader
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import io
from pythainlp import word_tokenize
from pythainlp.util import normalize
import pdfplumber  # Added for better Thai PDF extraction

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Initialize sentence transformer model for embeddings
# Using multilingual model for better Thai language support
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# MongoDB setup
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['chat_history']
sessions = db['sessions']
chats = db['conversations']
embeddings = db['embeddings'] 

# Streamlit page config
st.set_page_config(
    page_title="AI Chat Assistant with RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session states
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.current_session_id = None
    st.session_state.chat_history = []
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}
    st.session_state.chunk_size = 512
    st.session_state.chunk_overlap = 50

def normalize_thai_text(text):
    """Normalize Thai text by cleaning and standardizing characters."""
    text = normalize(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.]', '', text)
    return text.strip()

def chunk_thai_text(text, chunk_size=512, overlap=50):
    """Chunk Thai text efficiently."""
    text = normalize_thai_text(text)
    sentences = text.split('.')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = word_tokenize(sentence)
        sentence_length = len(words)
        
        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
            
            if chunks and overlap > 0:
                last_chunk_words = word_tokenize(chunks[-1])
                overlap_words = last_chunk_words[-overlap:]
                current_chunk = overlap_words + words
            else:
                current_chunk = words
                
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += sentence_length
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def clear_pdfs():
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}
    # Clear embeddings for the current session
    if st.session_state.current_session_id:
        embeddings.delete_many({'session_id': ObjectId(st.session_state.current_session_id)})

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        # If this is not the first chunk, start 'overlap' tokens back
        if start > 0:
            start = start - overlap
        chunk = text[start:end]
        # Clean chunk
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if chunk:
            chunks.append(chunk)
        start = end
    
    return chunks

def compute_embeddings(text_chunks, session_id, filename):
    """Compute embeddings for text chunks and store in MongoDB."""
    for i, chunk in enumerate(text_chunks):
        embedding = embedder.encode(chunk, convert_to_tensor=True)
        embedding_list = embedding.cpu().numpy().tolist()
        
        embeddings.insert_one({
            'session_id': ObjectId(session_id),
            'filename': filename,
            'chunk_index': i,
            'text': chunk,
            'embedding': embedding_list
        })

def retrieve_relevant_chunks(query, top_k=1):
    """Retrieve the most relevant chunks for the given query."""
    if not st.session_state.current_session_id:
        return []
    
    try:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        stored_embeddings = list(embeddings.find({'session_id': ObjectId(st.session_state.current_session_id)}).limit(10))
        
        if not stored_embeddings:
            return []
        
        similarities = []
        for doc in stored_embeddings:
            doc_embedding = torch.tensor(doc['embedding'])
            similarity = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), doc_embedding.cpu().numpy().reshape(1, -1))[0][0]
            similarities.append((similarity, doc['text'], doc['filename']))
        
        similarities.sort(reverse=True)
        return similarities[:top_k]
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return []

def create_new_session():
    session = {
        'created_at': datetime.now(),
        'name': f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    }
    session_id = sessions.insert_one(session).inserted_id
    # สร้างเอกสารในคอลเลกชัน `chats` สำหรับ session_id นี้
    chats.insert_one({
        'session_id': session_id,
        'messages': []  # เริ่มต้นด้วยลิสต์ว่าง
    })
    st.session_state.current_session_id = str(session_id)
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}
    return session_id

def save_to_mongodb(user_message, ai_response):
    if not st.session_state.current_session_id:
        create_new_session()
    # เพิ่มข้อความใหม่ลงในฟิลด์ `messages` ของ session_id ที่กำหนด
    chats.update_one(
        {'session_id': ObjectId(st.session_state.current_session_id)},  # ค้นหาเอกสารที่ตรงกับ session_id
        {
            '$push': {
                'messages': {
                    'timestamp': datetime.now(),
                    'user_message': user_message,
                    'ai_response': ai_response
                }
            }
        },
        upsert=True  # หากไม่มีเอกสาร ให้สร้างเอกสารใหม่
    )

def load_session_messages(session_id):
    session_chat = chats.find_one({'session_id': ObjectId(session_id)})
    messages = []
    chat_history = []
    if session_chat and 'messages' in session_chat:
        for chat in session_chat['messages']:
            messages.extend([
                {"role": "user", "content": chat['user_message']},
                {"role": "assistant", "content": chat['ai_response']}
            ])
            chat_history.append({"user": chat['user_message'], "assistant": chat['ai_response']})
    return messages, chat_history

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
                    st.warning(f"Warning: Fallback extraction failed for a page: {str(e)}")
                    continue
        
        full_text = "\n".join(text)
        full_text = normalize_thai_text(full_text)
        
        return full_text if full_text else None
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_chat_response(prompt, history, pdf_contents=None):
    try:
        # Retrieve relevant chunks based on the query
        relevant_chunks = retrieve_relevant_chunks(prompt)
        
        # Create a multilingual context
        system_prompt = (
            "You are a helpful AI assistant capable of understanding and responding in multiple languages "
            "including Thai and English. Answer based on the provided context if available, "
            "or use your general knowledge if needed. Keep responses clear and concise. "
            "Respond in the same language as the user's query."
        )
        
        # Prepare context with length limits and proper encoding
        context = ""
        if relevant_chunks:
            # Take only top 2 most relevant chunks to avoid context length issues
            for similarity, chunk, filename in relevant_chunks[:2]:
                # Ensure proper encoding of Thai characters
                chunk = chunk.encode('utf-8').decode('utf-8')
                if len(chunk) > 500:  # Limit chunk size
                    chunk = chunk[:500] + "..."
                context += f"\nContext from {filename}: {chunk}\n"
        
        # Prepare recent history with proper encoding
        recent_history = ""
        if history:
            last_exchange = history[-1]
            user_msg = last_exchange['user'].encode('utf-8').decode('utf-8')
            assistant_msg = last_exchange['assistant'].encode('utf-8').decode('utf-8')
            recent_history = f"Last exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n"
        
        # Ensure prompt is properly encoded
        encoded_prompt = prompt.encode('utf-8').decode('utf-8')
        
        # Combine all parts with controlled length
        conversation = f"{system_prompt}\n\n{context}\n\n{recent_history}\n\nUser: {encoded_prompt}\nAssistant:"
        
        # Generate response with more permissive settings
        try:
            response = model.generate_content(
                conversation,
                generation_config={
                    'temperature': 0.7,  # More creative responses
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 1024,  # Longer responses for Thai language
                    'candidate_count': 1
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            )
            
            if hasattr(response, 'text') and response.text:
                # Ensure the response is properly encoded
                return response.text.strip().encode('utf-8').decode('utf-8')
            else:
                return "ขออภัย ไม่สามารถสร้างคำตอบได้ กรุณาลองถามใหม่อีกครั้ง" if is_thai(prompt) else \
                       "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"Response generation error: {error_msg}")
            
            # Bilingual error messages
            if "500" in error_msg:
                return "ขออภัย คำถามอาจจะซับซ้อนเกินไป กรุณาแบ่งคำถามเป็นส่วนย่อยๆ" if is_thai(prompt) else \
                       "I apologize, but your question might be too complex. Please try breaking it down into simpler parts."
            elif "429" in error_msg:
                return "ขณะนี้มีการร้องขอมากเกินไป กรุณารอสักครู่แล้วลองใหม่" if is_thai(prompt) else \
                       "I'm receiving too many requests right now. Please wait a moment and try again."
            elif "safety" in error_msg.lower() or "harm" in error_msg.lower():
                return "ขออภัย ไม่สามารถตอบคำถามนี้ได้เนื่องจากข้อจำกัดด้านความปลอดภัย กรุณาถามใหม่" if is_thai(prompt) else \
                       "I apologize, but I cannot generate a response to this query due to content safety restrictions. Please try rephrasing your question."
            else:
                return "เกิดข้อผิดพลาด กรุณาลองใหม่หรือเริ่มเซสชันใหม่" if is_thai(prompt) else \
                       "I encountered an error. Please try rephrasing your question or starting a new session."
            
    except Exception as e:
        st.error(f"Chat processing error: {str(e)}")
        return "เกิดข้อผิดพลาดในการประมวลผลแชท กรุณาลองใหม่" if is_thai(prompt) else \
               "Something went wrong with the chat processing. Please try again."

def is_thai(text):
    """Check if the text contains Thai characters."""
    thai_chars = set('\u0E00-\u0E7F')
    return any((char in thai_chars) for char in text)

# Main UI
st.title("🤖 RAG-Enhanced AI Chat Assistant")

# Sidebar for session management and chat history
with st.sidebar:
    st.header("Chat Sessions")

    tab_main, tab_advanced = st.tabs(["Main", "Settings"])
    
    with tab_main:
        # New Session Button
        if st.button("New Chat Session"):
            create_new_session()
            st.experimental_rerun()

        # PDF Upload
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_key = uploaded_file.name
                
                # Only process new or changed files
                if (file_key not in st.session_state.uploaded_files or 
                    uploaded_file != st.session_state.uploaded_files[file_key]):
                    
                    with st.spinner(f"Processing {file_key}..."):
                        # Read file content
                        file_content = uploaded_file.read()
                        
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(file_content)
                        
                        if pdf_text:
                            st.session_state.pdf_contents[file_key] = pdf_text
                            st.session_state.uploaded_files[file_key] = uploaded_file
                            
                            # Process text into chunks and compute embeddings
                            if st.session_state.current_session_id:
                                chunks = chunk_thai_text(
                                    pdf_text, chunk_size=st.session_state.chunk_size, 
                                    overlap=st.session_state.chunk_overlap
                                )

                                compute_embeddings(
                                    chunks,
                                    st.session_state.current_session_id,
                                    file_key
                                )
                        else:
                            st.error(f"Failed to extract text from {file_key}")
            
            if st.session_state.pdf_contents:
                st.success(f"Successfully processed {len(st.session_state.pdf_contents)} PDF file(s)")
                
                # Display uploaded files
                st.write("Uploaded PDFs:")
                for filename in st.session_state.pdf_contents.keys():
                    st.write(f"📄 {filename}")
        
        # Show clear button only if there are PDFs
        if st.session_state.pdf_contents:
            if st.button("Clear All PDFs", key="clear_pdfs"):
                clear_pdfs()
                st.experimental_rerun()
    
        # Session selector
        all_sessions = list(sessions.find().sort('created_at', -1))
        if all_sessions:
            st.subheader("Previous Sessions")
            for session in all_sessions:
                session_id = str(session['_id'])
                if st.button(f"📝 {session['name']}", key=session_id):
                    st.session_state.current_session_id = session_id
                    st.session_state.messages, st.session_state.chat_history = load_session_messages(session_id)
                    st.experimental_rerun()
        
        # Clear all history button
        if st.button("Clear All History"):
            sessions.delete_many({})
            chats.delete_many({})
            embeddings.delete_many({})  # Also clear embeddings
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.rerun()

    with tab_advanced:
        st.subheader("RAG Settings")
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=128,
            max_value=1024,
            value=512,
            step=64,
            help="Size of text chunks for processing documents"
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=256,
            value=50,
            step=16,
            help="Overlap between consecutive chunks"
        )
        
        st.subheader("Custom Prompt")
        custom_prompt = st.text_area(
            "Set a custom prompt to guide AI responses:",
            placeholder="e.g., You are a helpful assistant specialized in IT security."
        )

# Display current session info
if st.session_state.current_session_id:
    session = sessions.find_one({'_id': ObjectId(st.session_state.current_session_id)})
    st.info(f"Current Session: {session['name']}")
else:
    st.info("No active session. Click 'New Chat Session' to start chatting.")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if st.session_state.current_session_id:
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = get_chat_response(prompt, st.session_state.chat_history, st.session_state.pdf_contents)
                st.markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.append({
                    "user": prompt,
                    "assistant": response_text
                })
                
                # Save to session state and MongoDB
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                save_to_mongodb(prompt, response_text)
else:
    st.write("👆 Create a new session to start chatting!")