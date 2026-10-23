
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_FILE)


class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    CONTEXT_TOKEN_BUDGET: int = int(
    os.getenv("CONTEXT_TOKEN_BUDGET", "12000")
    )

    CONTEXT_SAFETY_MARGIN: int = int(
        os.getenv("CONTEXT_SAFETY_MARGIN", "1000")
    )

    MAX_OUTPUT_TOKENS: int = int(
        os.getenv("MAX_OUTPUT_TOKENS", "2048")
    )


settings = Settings()