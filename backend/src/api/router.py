from fastapi import APIRouter
from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as document_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(document_router)