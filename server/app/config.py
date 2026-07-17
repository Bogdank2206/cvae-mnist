from pathlib import Path

# Корневая директория сервера — от неё отсчитываем пути к весам и логам
BASE_DIR = Path(__file__).resolve().parent.parent

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
