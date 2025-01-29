import streamlit as st
import google.generativeai as genai
import os
import tempfile
from PyPDF2 import PdfReader
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
# Comment out unused imports
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import torch
import re
import io
from pythainlp import word_tokenize
from pythainlp.util import normalize
import pdfplumber

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')
# Remove vision model if not used
# vision_model = genai.GenerativeModel('gemini-pro-vision')

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
        # Compute embedding
        embedding = embedder.encode(chunk, convert_to_tensor=True)
        
        # Convert embedding to list for MongoDB storage
        embedding_list = embedding.cpu().numpy().tolist()
        
        # Store in MongoDB
        embeddings.insert_one({
            'session_id': ObjectId(session_id),
            'filename': filename,
            'chunk_index': i,
            'text': chunk,
            'embedding': embedding_list
        })


def retrieve_relevant_chunks(query, session_id):
    """Retrieve relevant chunks using semantic matching"""
    try:
        # Generate a search query representation
        query_response = model.generate_content(
            f"Please provide a concise summary of this query: {query}",
            generation_config={
                'temperature': 0.0,
                'candidate_count': 1,
                'max_output_tokens': 1024,
            }
        )
        query_summary = query_response.text

        # Retrieve stored documents
        stored_docs = list(embeddings.find({'session_id': ObjectId(session_id)}))
        
        if not stored_docs:
            return []

        # Compare query with stored chunks
        relevant_chunks = []
        for doc in stored_docs:
            try:
                # Compare summaries
                comparison_response = model.generate_content(
                    f"""Compare these two text summaries and rate their similarity from 0 to 1:
                    Text 1: {query_summary}
                    Text 2: {doc['embedding']}
                    Only respond with a number between 0 and 1.""",
                    generation_config={
                        'temperature': 0.0,
                        'candidate_count': 1,
                    }
                )
                
                try:
                    similarity = float(comparison_response.text.strip())
                except:
                    similarity = 0.0
                
                relevant_chunks.append((
                    similarity,
                    doc['text'],
                    doc['filename']
                ))
            except Exception as e:
                continue

        # Sort by similarity and return top chunks
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        return relevant_chunks[:2]  # Return top 2 most similar chunks

    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
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
        # Check if prompt is empty
        if not prompt:
            return "กรุณาส่งคำถาม" if is_thai(prompt) else "Please submit a query"
        
        # Initialize context and system prompt
        context = ""
        system_prompt = (
            "You are an AI personal assistant with the ability to interact with users naturally and provide helpful," \
            "context-aware responses. Your task is to assist users by understanding and responding to their questions based on the information available to you." \
            "You are also capable of reading and processing PDF documents to gather relevant information to provide accurate answers. When a user uploads a PDF file, " \
            "you will extract the content and use it to inform your responses, ensuring that your answers are accurate and based on the document's information. " \
            "Your responses should be clear, concise, and user-friendly, reflecting your role as a helpful assistant.\n\n" 
        )
        
        # If PDF contents exist, use RAG approach
        if pdf_contents and st.session_state.current_session_id:
            # Retrieve relevant chunks using embeddings
            relevant_chunks = retrieve_relevant_chunks(prompt, st.session_state.current_session_id)
            
            if relevant_chunks:
                # Modify system prompt for RAG
                system_prompt += (
                    " Base your response primarily on the provided context, "
                    "while maintaining a natural conversational flow."
                )
                
                # Add context from relevant chunks
                for similarity, chunk, filename in relevant_chunks[:2]:
                    chunk = chunk.encode('utf-8').decode('utf-8')
                    if len(chunk) > 500:
                        chunk = chunk[:500] + "..."
                    context += f"\nContext from {filename} (similarity: {similarity:.2f}):\n{chunk}\n"
        
        # Add recent conversation history if available
        recent_history = ""
        if history:
            last_exchange = history[-1]
            user_msg = last_exchange['user'].encode('utf-8').decode('utf-8')
            assistant_msg = last_exchange['assistant'].encode('utf-8').decode('utf-8')
            recent_history = f"Previous exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n"
        
        # Construct the final conversation prompt
        encoded_prompt = prompt.encode('utf-8').decode('utf-8')
        conversation = f"{system_prompt}\n\n"
        
        if context:
            conversation += f"Reference context:\n{context}\n\n"
        
        if recent_history:
            conversation += f"{recent_history}\n"
            
        conversation += f"User: {encoded_prompt}\nAssistant:"
        
        # Generate response using Gemini
        try:
            response = model.generate_content(
                conversation,
                generation_config={
                    'temperature': 0.5,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                    'candidate_count': 1
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                return "ขออภัย ไม่สามารถสร้างคำตอบได้ กรุณาลองถามใหม่อีกครั้ง" if is_thai(prompt) else \
                       "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                       
        except Exception as api_error:
            st.error(f"Error with Gemini API: {str(api_error)}")
            return "เกิดข้อผิดพลาดในการติดต่อ API กรุณาลองใหม่" if is_thai(prompt) else \
                   "An error occurred while contacting the API. Please try again."

    except Exception as e:
        st.error(f"Chat processing error: {str(e)}")
        return "เกิดข้อผิดพลาดในการประมวลผลแชท กรุณาลองใหม่" if is_thai(prompt) else \
               "Something went wrong with the chat processing. Please try again."

def normalize_thai_text(text):
    """Normalize Thai text by cleaning and standardizing characters."""
    text = normalize(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.]', '', text)
    return text.strip()

# เพิ่มฟังก์ชั่น chunk_thai_text ที่ขาดหายไป
def chunk_thai_text(text, chunk_size=512, overlap=50):
    """ปรับปรุงการแบ่งข้อความภาษาไทยเป็นส่วนๆ"""
    # Normalize text
    text = normalize_thai_text(text)
    
    # Split text into sentences first
    sentences = text.split('.')
    
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
                chunk_text = ' '.join(current_chunk)
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
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def is_thai(text):
    """Check if the text contains Thai characters."""
    if not text:  # Handle None or empty string
        return False
    thai_pattern = re.compile('[\u0E00-\u0E7F]')
    return bool(thai_pattern.search(text))


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