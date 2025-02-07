import google.generativeai as genai
import os
import streamlit as st
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
from db_handler import db_handler

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# โหลด SentenceTransformer (ใช้โมเดลที่ดีสำหรับภาษาไทย)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def retrieve_relevant_chunks(query, session_id):
    """Retrieve relevant chunks using semantic similarity"""
    try:
        # 🔹 ตรวจสอบว่า session_id เป็น ObjectId หรือยังเป็น string
        if not isinstance(session_id, ObjectId):
            session_id = ObjectId(session_id)

        # 🔹 ดึงข้อมูล embeddings จาก MongoDB
        stored_docs = list(db_handler().embeddings.find({"session_id": session_id}))
        if not stored_docs:
            return []

        # 🔹 คำนวณ embedding ของ Query
        query_embedding = embedding_model.encode([query])

        # 🔹 คำนวณ cosine similarity กับทุก chunk ในฐานข้อมูล
        relevant_chunks = []
        for doc in stored_docs:
            try:
                # ดึงค่า embedding ของเอกสารจาก MongoDB
                doc_embedding = np.array(doc["embedding"])  # แปลงเป็น NumPy array
                similarity = cosine_similarity(query_embedding, [doc_embedding])[0][0]

                # เพิ่มลงในรายการ
                relevant_chunks.append((similarity, doc["text"], doc["filename"]))
            except Exception as e:
                continue  # ถ้าคำนวณผิดพลาด ข้ามไป

        # 🔹 เรียงข้อมูลตามค่า similarity (จากมากไปน้อย)
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)

        return relevant_chunks[:2]  # ส่งคืน top 2 chunk ที่คล้ายที่สุด

    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
        return []


def get_chat_response(prompt, history, pdf_contents=None):
    try:
        # 🟢 ตรวจสอบว่ามี PDF หรือไม่
        has_pdf = pdf_contents and st.session_state.current_session_id

        # 🟢 ตั้งค่าบทบาทของ LLM
        system_prompt = (
            "You are an AI assistant capable of answering general knowledge questions. "
            "Respond to the following message using the same language and tone."
            "However, if a PDF document is provided, you should prioritize answering based on its content when relevant. "
            "If the user's question is unrelated to the PDF, you may answer using your general knowledge.\n\n"
        )

        context = ""

        # 🟢 กรณีมี PDF → ใช้ RAG (Retrieval-Augmented Generation)
        if has_pdf:
            relevant_chunks = retrieve_relevant_chunks(
                prompt, st.session_state.current_session_id
            )

            if relevant_chunks:
                system_prompt += "Please base your response primarily on the provided reference context.\n\n"

                # 🔹 เพิ่ม context จาก PDF
                for similarity, chunk, filename in relevant_chunks[:2]:
                    chunk = chunk.encode("utf-8").decode("utf-8")
                    context += f"\n[Context from {filename} (similarity: {similarity:.2f})]\n{chunk}\n"

            # 🟢 กรณีไม่มี PDF → ให้ LLM ตอบคำถามทั่วไป
        else:
            system_prompt += "The question is unrelated to any uploaded documents, so you may answer normally.\n\n"

        # 🔹 ดึงข้อความล่าสุดจากประวัติการสนทนา (ถ้ามี)
        recent_history = ""
        if history:
            last_exchange = history[-1]
            user_msg = last_exchange["user"].encode("utf-8").decode("utf-8")
            assistant_msg = last_exchange["assistant"].encode("utf-8").decode("utf-8")
            recent_history = (
                f"Previous exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n"
            )

        # Construct the final conversation prompt
        conversation = f"{system_prompt}\n\n"

        if context:
            conversation += f"Reference context:\n{context}\n\n"

        if recent_history:
            conversation += f"{recent_history}\n"

        conversation += f"User: {prompt.encode('utf-8').decode('utf-8')}\nAssistant:"

        # Generate response using Gemini
        response = model.generate_content(
            conversation,
            generation_config={
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
                "candidate_count": 1,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ],
        )

        # 🟢 ตรวจสอบและส่งคืนผลลัพธ์
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "I couldn't generate a proper response. Please try again."

    except Exception as e:
        st.error(f"Chat processing error: {str(e)}")
        return "Something went wrong. Please try again."


# def is_thai(text):
#     """Check if the text contains Thai characters."""
#     if not text:  # Handle None or empty string
#         return False
#     thai_pattern = re.compile("[\u0E00-\u0E7F]")
#     return bool(thai_pattern.search(text))


# def clear_pdfs():
#     st.session_state.pdf_contents = {}
#     st.session_state.uploaded_files = {}
#     # Clear embeddings for the current session
#     if st.session_state.current_session_id:
#         embeddings.delete_many(
#             {"session_id": ObjectId(st.session_state.current_session_id)}
#         )
