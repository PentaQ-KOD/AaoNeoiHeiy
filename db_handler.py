from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import streamlit as st
import os


class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["chat_history"]
        self.sessions = self.db["sessions"]
        self.chats = self.db["conversations"]
        self.embeddings = self.db["embeddings"]  # คอลเลกชัน embeddings


def db_handler():
    return MongoDBHandler()


def create_new_session():
    db = db_handler()  # สร้างอินสแตนซ์ของ MongoDBHandler
    session = {
        "created_at": datetime.now(),
        "name": f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    }
    session_id = db.sessions.insert_one(session).inserted_id  # ใช้ db.sessions
    # สร้างเอกสารในคอลเลกชัน `chats` สำหรับ session_id นี้
    db.chats.insert_one({"session_id": session_id, "messages": []})  # เริ่มต้นด้วยลิสต์ว่าง
    st.session_state.current_session_id = str(session_id)
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.pdf_contents = {}
    st.session_state.uploaded_files = {}
    return session_id


def save_to_mongodb(user_message, ai_response):
    if not st.session_state.current_session_id:
        create_new_session()
    db = db_handler()  # สร้างอินสแตนซ์ของ MongoDBHandler
    # เพิ่มข้อความใหม่ลงในฟิลด์ `messages` ของ session_id ที่กำหนด
    db.chats.update_one(
        {
            "session_id": ObjectId(
                st.session_state.current_session_id
            )  # ใช้ ObjectId จาก bson
        },
        {
            "$push": {
                "messages": {
                    "timestamp": datetime.now(),
                    "user_message": user_message,
                    "ai_response": ai_response,
                }
            }
        },
        upsert=True,
    )


def load_session_messages(session_id):
    db = db_handler()  # สร้างอินสแตนซ์ของ MongoDBHandler
    session_chat = db.chats.find_one({"session_id": ObjectId(session_id)})
    messages = []
    chat_history = []
    if session_chat and "messages" in session_chat:
        for chat in session_chat["messages"]:
            messages.extend(
                [
                    {"role": "user", "content": chat["user_message"]},
                    {"role": "assistant", "content": chat["ai_response"]},
                ]
            )
            chat_history.append(
                {"user": chat["user_message"], "assistant": chat["ai_response"]}
            )
    return messages, chat_history
