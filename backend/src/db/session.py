from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

from sqlalchemy import text

from src.core.config import settings

if not settings.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from configuration")

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal =async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def check_db_connection() -> bool:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return True