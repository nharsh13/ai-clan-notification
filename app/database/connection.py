from sqlalchemy import create_engine

from app.config import DATABASE_URL

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be configured")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL)