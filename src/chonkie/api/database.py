"""Database configuration and session management."""

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/chonkie.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session, for use as a FastAPI dependency."""
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    """Create all SQLAlchemy-mapped tables if they do not already exist."""
    # Ensure the directory for the SQLite database file exists.
    db_url = DATABASE_URL
    if db_url.startswith("sqlite"):
        # Extract the file path from the URL (strip the dialect prefix and slashes)
        path = db_url.split("///", 1)[-1]
        if path and path not in (":memory:", ""):
            db_dir = os.path.dirname(path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
