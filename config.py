import os
from dotenv import load_dotenv


class Config:
    @staticmethod
    def load_env():
        load_dotenv()  # โหลดไฟล์ .env
        google_api_key = os.getenv("GOOGLE_API_KEY")
        mongodb_uri = os.getenv("MONGODB_URI")

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in .env file")

        os.environ["GOOGLE_API_KEY"] = google_api_key
        os.environ["MONGODB_URI"] = mongodb_uri
