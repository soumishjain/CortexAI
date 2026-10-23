from fastapi import FastAPI, HTTPException
from src.core.config import settings
from src.api.router import api_router

if not settings.GOOGLE_API_KEY:
    raise ValueError("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")

if not settings.GEMINI_MODEL:
    raise ValueError("Gemini model is not set. Please set the GEMINI_MODEL environment variable.")

app = FastAPI(title="CortexAI API")
app.include_router(api_router)