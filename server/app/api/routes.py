import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services.inference import generate_digit

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Генерирует изображение заданной цифры."""
    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(None, generate_digit, req.label)
    if image is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена — запустите обучение",
        )
    return GenerateResponse(image=image)
