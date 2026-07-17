from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Запрос на генерацию цифры."""

    label: int = Field(..., ge=0, le=9, description="Лейбл цифры (0–9)")


class GenerateResponse(BaseModel):
    """Ответ с base64-кодированным PNG-изображением."""

    image: str = Field(..., description="data:image/png;base64,...")
