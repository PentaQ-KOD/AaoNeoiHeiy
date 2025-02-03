import google.generativeai as genai
import os
import streamlit as st
import re
from bson import ObjectId
from db_handler import db_handler

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
# vision_model = genai.GenerativeModel('gemini-pro-vision')


def retrieve_relevant_chunks(query, session_id):
    """Retrieve relevant chunks using semantic matching"""
    try:
        # Generate a search query representation
        query_response = model.generate_content(
            f"Please provide a concise summary of this query: {query}",
            generation_config={
                "temperature": 0.0,
                "candidate_count": 1,
                "max_output_tokens": 256,
            },
        )
        query_summary = query_response.text

        # Retrieve stored documents
        # stored_docs = list(embeddings.find({"session_id": ObjectId(session_id)}))
        stored_docs = list(
            db_handler().embeddings.find({"session_id": ObjectId(session_id)})
        )

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
                        "temperature": 0.0,
                        "candidate_count": 1,
                    },
                )

                try:
                    similarity = float(comparison_response.text.strip())
                except:
                    similarity = 0.0

                relevant_chunks.append((similarity, doc["text"], doc["filename"]))
            except Exception as e:
                continue

        # Sort by similarity and return top chunks
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        return relevant_chunks[:2]  # Return top 2 most similar chunks

    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
        return []


def get_chat_response(prompt, history, pdf_contents=None):
    try:
        # Check if prompt is empty
        if not prompt:
            return "กรุณาส่งคำถาม" if is_thai(prompt) else "Please submit a query"

        # Initialize context and system prompt
        context = ""
        system_prompt = (
            "You are an AI personal assistant with the ability to interact with users naturally and provide helpful,"
            "context-aware responses. Your task is to assist users by understanding and responding to their questions based on the information available to you."
            "You are also capable of reading and processing PDF documents to gather relevant information to provide accurate answers. When a user uploads a PDF file, "
            "you will extract the content and use it to inform your responses, ensuring that your answers are accurate and based on the document's information. "
            "Your responses should be clear, concise, and user-friendly, reflecting your role as a helpful assistant.\n\n"
        )

        # If PDF contents exist, use RAG approach
        if pdf_contents and st.session_state.current_session_id:
            # Retrieve relevant chunks using embeddings
            relevant_chunks = retrieve_relevant_chunks(
                prompt, st.session_state.current_session_id
            )

            if relevant_chunks:
                # Modify system prompt for RAG
                system_prompt += (
                    " Base your response primarily on the provided context, "
                    "while maintaining a natural conversational flow."
                )

                # Add context from relevant chunks
                for similarity, chunk, filename in relevant_chunks[:2]:
                    chunk = chunk.encode("utf-8").decode("utf-8")
                    if len(chunk) > 800:
                        chunk = chunk[:800] + "..."
                    context += f"\nContext from {filename} (similarity: {similarity:.2f}):\n{chunk}\n"

        # Add recent conversation history if available
        recent_history = ""
        if history:
            last_exchange = history[-1]
            user_msg = last_exchange["user"].encode("utf-8").decode("utf-8")
            assistant_msg = last_exchange["assistant"].encode("utf-8").decode("utf-8")
            recent_history = (
                f"Previous exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n"
            )

        # Construct the final conversation prompt
        encoded_prompt = prompt.encode("utf-8").decode("utf-8")
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
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 1024,
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

            if hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                return (
                    "ขออภัย ไม่สามารถสร้างคำตอบได้ กรุณาลองถามใหม่อีกครั้ง"
                    if is_thai(prompt)
                    else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                )

        except Exception as api_error:
            st.error(f"Error with Gemini API: {str(api_error)}")
            return (
                "เกิดข้อผิดพลาดในการติดต่อ API กรุณาลองใหม่"
                if is_thai(prompt)
                else "An error occurred while contacting the API. Please try again."
            )

    except Exception as e:
        st.error(f"Chat processing error: {str(e)}")
        return (
            "เกิดข้อผิดพลาดในการประมวลผลแชท กรุณาลองใหม่"
            if is_thai(prompt)
            else "Something went wrong with the chat processing. Please try again."
        )


def is_thai(text):
    """Check if the text contains Thai characters."""
    if not text:  # Handle None or empty string
        return False
    thai_pattern = re.compile("[\u0E00-\u0E7F]")
    return bool(thai_pattern.search(text))


# def clear_pdfs():
#     st.session_state.pdf_contents = {}
#     st.session_state.uploaded_files = {}
#     # Clear embeddings for the current session
#     if st.session_state.current_session_id:
#         embeddings.delete_many(
#             {"session_id": ObjectId(st.session_state.current_session_id)}
#         )
