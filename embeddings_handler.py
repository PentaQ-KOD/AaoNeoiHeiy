from sentence_transformers import SentenceTransformer
from bson import ObjectId
import re
from db_handler import db_handler

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class EmbeddingsHandler:
    def __init__(self):
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def compute_embeddings(text_chunks, session_id, filename):
    """Compute embeddings for text chunks and store in MongoDB."""
    for i, chunk in enumerate(text_chunks):
        # Compute embedding
        embedding = embedder.encode(chunk, convert_to_tensor=True)

        # Convert embedding to list for MongoDB storage
        embedding_list = embedding.cpu().numpy().tolist()

        # Store in MongoDB
        db_handler().embeddings.insert_one(
            {
                "session_id": ObjectId(session_id),
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding_list,
            }
        )


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
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks
