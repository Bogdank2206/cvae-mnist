import logging
from pathlib import Path

import torch

from app.config import LATENT_DIM, MODEL_PATH, PRIOR_VAR
from app.models.cvae import CVAE
from app.services.image import tensor_to_base64_png

logger = logging.getLogger(__name__)

_model: CVAE | None = None
_device: torch.device = torch.device("cpu")


def load_model() -> None:
    """Загружает веса модели в память (однократно при старте)."""
    global _model
    if _model is not None:
        return
    if not MODEL_PATH.exists():
        logger.warning("Файл весов %s не найден — генерация невозможна", MODEL_PATH)
        return
    _model = CVAE(latent_dim=LATENT_DIM, prior_var=PRIOR_VAR)
    state = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()
    logger.info("Модель загружена из %s", MODEL_PATH)


def generate_digit(label: int) -> str | None:
    """Генерирует изображение цифры и возвращает base64 PNG data-URI."""
    if _model is None:
        return None
    with torch.no_grad():
        label_tensor = torch.tensor([label], dtype=torch.long, device=_device)
        image: torch.Tensor = _model.generate(label_tensor, device=_device)
    return tensor_to_base64_png(image)
