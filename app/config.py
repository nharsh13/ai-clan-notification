import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
REMOTE_NOTIFICATION_SEND_URL = os.getenv("REMOTE_NOTIFICATION_SEND_URL", "")
REMOTE_NOTIFICATION_TIMEOUT_SECONDS = float(os.getenv("REMOTE_NOTIFICATION_TIMEOUT_SECONDS", "30"))
