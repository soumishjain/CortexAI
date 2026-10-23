import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from src.db.base import Base
import src.db.models  # Register all models


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

if not TEST_DATABASE_URL:
    raise RuntimeError("TEST_DATABASE_URL is not configured.")

if not TEST_DATABASE_URL.endswith("/cortexai_test"):
    raise RuntimeError(
        "Tests are allowed to use only the cortexai_test database."
    )


@pytest.fixture
async def db_session():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        pool_pre_ping=True,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)

    await engine.dispose()