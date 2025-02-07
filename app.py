import streamlit as st
from bson import ObjectId

from config import Config
from pdf_handler import extract_text_from_pdf, chunk_thai_text
from db_handler import (
    create_new_session,
    save_to_mongodb,
    load_session_messages,
    db_handler,
)
from chat_handler import get_chat_response, retrieve_relevant_chunks
from embeddings_handler import EmbeddingsHandler, chunk_text, compute_embeddings
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables
Config.load_env()


# Streamlit page config
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# Initialize session states
# embeddings = get_embeddings_collection()
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.current_session_id = None
    st.session_state.chat_history = []
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}
    st.session_state.chunk_size = 800
    st.session_state.chunk_overlap = 80


def clear_pdfs():
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}


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
            clear_pdfs()
            st.experimental_rerun()

        # ตรวจสอบว่า session_state มี key ที่ใช้เก็บไฟล์หรือไม่
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
        if "pdf_contents" not in st.session_state:
            st.session_state.pdf_contents = {}

        # PDF Upload
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        # ตรวจหาไฟล์ที่ถูกลบออก
        existing_files = set(st.session_state.uploaded_files.keys())
        current_files = (
            {file.name for file in uploaded_files} if uploaded_files else set()
        )
        deleted_files = existing_files - current_files  # หาชื่อไฟล์ที่ถูกลบไป

        # ลบไฟล์ที่ไม่อยู่ใน uploaded_files แล้ว
        for deleted_file in deleted_files:
            del st.session_state.uploaded_files[deleted_file]
            if deleted_file in st.session_state.pdf_contents:
                del st.session_state.pdf_contents[deleted_file]

            st.experimental_rerun()  # รีเฟรช UI

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_key = uploaded_file.name

                # Only process new or changed files
                if (
                    file_key not in st.session_state.uploaded_files
                    or uploaded_file != st.session_state.uploaded_files[file_key]
                ):

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
                                    pdf_text,
                                    chunk_size=st.session_state.chunk_size,
                                    overlap=st.session_state.chunk_overlap,
                                )

                                compute_embeddings(
                                    chunks,
                                    st.session_state.current_session_id,
                                    file_key,
                                )
                        else:
                            st.error(f"Failed to extract text from {file_key}")

            if st.session_state.pdf_contents:
                st.success(
                    f"Successfully processed {len(st.session_state.pdf_contents)} PDF file(s)"
                )

        # Session selector
        all_sessions = list(db_handler().sessions.find().sort("created_at", -1))
        if all_sessions:
            st.subheader("Previous Sessions")
            for session in all_sessions:
                session_id = str(session["_id"])
                if st.button(f"📝 {session['name']}", key=session_id):
                    st.session_state.current_session_id = session_id
                    st.session_state.messages, st.session_state.chat_history = (
                        load_session_messages(session_id)
                    )
                    st.experimental_rerun()

        # Clear all history button
        if st.button("Clear All History"):
            db_handler().sessions.delete_many({})
            db_handler().chats.delete_many({})
            db_handler().embeddings.delete_many({})  # Also clear embeddings
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
            value=800,
            step=64,
            help="Size of text chunks for processing documents",
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=256,
            value=80,
            step=16,
            help="Overlap between consecutive chunks",
        )

        st.subheader("Custom Prompt")
        custom_prompt = st.text_area(
            "Set a custom prompt to guide AI responses:",
            placeholder="e.g., You are a helpful assistant specialized in IT security.",
        )

# Display current session info
if st.session_state.current_session_id:
    session = db_handler().sessions.find_one(
        {"_id": ObjectId(st.session_state.current_session_id)}
    )
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
                response_text = get_chat_response(
                    prompt, st.session_state.chat_history, st.session_state.pdf_contents
                )
                st.markdown(response_text)

                # Update chat history
                st.session_state.chat_history.append(
                    {"user": prompt, "assistant": response_text}
                )

                # Save to session state and MongoDB
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
                save_to_mongodb(prompt, response_text)
else:
    st.write("👆 Create a new session to start chatting!")
