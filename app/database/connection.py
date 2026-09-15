from sqlalchemy import create_engine

from app.config import ConfigurationError, DATABASE_URL

if not DATABASE_URL:
    raise ConfigurationError(
        "Missing required environment variable: DATABASE_URL"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
