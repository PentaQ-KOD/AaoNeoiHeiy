import streamlit as st
import google.generativeai as genai
import os
import tempfile
from PyPDF2 import PdfReader
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId


# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('AIzaSyA3TMxcm4uSfNswb00RnzhRB5n4iWN9I3k'))
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# MongoDB setup
client = MongoClient("mongodb+srv://inpantawat22:1234@agents.ci2qy.mongodb.net/?retryWrites=true&w=majority")
db = client['chat_history']
sessions = db['sessions']
chats = db['conversations']

# Streamlit page config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.current_session_id = None
    st.session_state.chat_history = []
    st.session_state.pdf_contents = {}  # Dictionary to store multiple PDF contents
    st.session_state.uploaded_files = {}  # Dictionary to track uploaded files

# Initialize session states
#if 'messages' not in st.session_state:
#    st.session_state.messages = []
#if 'current_session_id' not in st.session_state:
#    st.session_state.current_session_id = None

def clear_pdfs():
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

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

def get_chat_response(prompt, history, pdf_contents=None):
    # Create a conversation context that Gemini can understand
    conversation = "You are a helpful and knowledgeable AI assistant. Be natural and engaging in your responses. " \
                  "Maintain context from the conversation history but keep responses focused and relevant.\n\n"
    
    # Add PDF content context if available
    if pdf_contents and len(pdf_contents) > 0:
        conversation += "Context from PDF documents:\n"
        for filename, content in pdf_contents.items():
            conversation += f"\nFrom {filename}:\n{content}\n"
        conversation += "\n"

    # Add conversation history if it exists
    if history:
        conversation += "Previous conversation:\n"
        for entry in history[-5:]:  # Only use last 5 exchanges to avoid context length issues
            conversation += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
    
    # Add the current prompt
    conversation += f"User: {prompt}\nAssistant: "
    
    try:
        response = model.generate_content(conversation)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error. Could you please rephrase your question?"

# Main UI
st.title("🤖 AI Chat Assistant")

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
                    
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Extract text from PDF
                    pdf_text = extract_text_from_pdf(tmp_file_path)
                    st.session_state.pdf_contents[file_key] = pdf_text
                    st.session_state.uploaded_files[file_key] = uploaded_file
                    
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)
            
            st.success(f"Successfully processed {len(uploaded_files)} PDF file(s)")
            
            # Display uploaded files
            if hasattr(st.session_state, 'pdf_contents') and st.session_state.pdf_contents:
                st.write("Uploaded PDFs:")
                for filename in st.session_state.pdf_contents.keys():
                    st.write(f"📄 {filename}")
        
        # Show clear button only if there are PDFs
        if hasattr(st.session_state, 'pdf_contents') and st.session_state.pdf_contents:
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
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_session_id = None
            st.rerun()

    with tab_advanced:
        custom_prompt = ""
        if tab_advanced:
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