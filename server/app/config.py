import os
from pathlib import Path

from dotenv import load_dotenv

# Корневая директория сервера — от неё отсчитываем пути к весам и логам
BASE_DIR = Path(__file__).resolve().parent.parent

# Корневой .env монорепо — источник переменных для локального запуска.
# В Docker значения приходят через environment; load_dotenv не перезаписывает
# уже заданные переменные, поэтому конфиг работает в обоих режимах.
load_dotenv(BASE_DIR.parent / ".env")

# Директория с чекпоинтами модели
WEIGHTS_DIR = BASE_DIR / "weights"

# Имя файла весов по умолчанию
MODEL_FILENAME = "cvae.pt"

# Полный путь к весам
MODEL_PATH = WEIGHTS_DIR / MODEL_FILENAME

# Размер латентного пространства — должен совпадать с моделью
LATENT_DIM: int = 32

# Дисперсия априорного p(z) = N(0, PRIOR_VAR · I): скаляр, единый для всех
# латентных размерностей (диагональная ковариация). Должен быть > 0 для log.
PRIOR_VAR: float = 1.0

# Хост и порт uvicorn (переопределяются через CLI-аргументы)
HOST: str = "0.0.0.0"
PORT: int = 8000

# Разрешённые CORS-origin'ы: список, т.к. dev (Vite :5173) и compose (nginx :80)
# дают разные origin. Парсится из запятой-разделённой строки в .env.
CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
