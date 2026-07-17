# CVAE — Генерация рукописных цифр (MNIST)

## Контекст проекта

Full-stack приложение: условный вариационный автокодировщик (CVAE), обученный на MNIST.
Frontend — интерактивный UI для генерации и визуализации цифр.
Backend — API для инференса модели и обучения.

| Слой    | Стек                                                     |
| ------- | -------------------------------------------------------- |
| Front   | React + Vite + TypeScript + Tailwind CSS + Framer Motion |
| Backend | FastAPI + PyTorch + Uvicorn                              |
| Модель  | CVAE (Conditional VAE), MNIST                            |
| Данные  | MNIST из Hugging Face (`datasets`)                       |

---

## Архитектура модели (CVAE)

Условный вариационный автокодировщик: генерация цифры conditioned на лейбл.

### Распределения

- **Приор**: $p(z) = \mathcal{N}(0, I)$
- **Лайклихуд**: $p(x \mid z) = \mathcal{N}(\mathrm{Decoder}(z), \sigma^2)$
- **Апостериорный аналог**: $q(z \mid x) = \mathcal{N}(\mu(x), \mathrm{diag}(\exp(\log\sigma^{2}(x))))$
- **Loss**: $-\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \| p(z))$ (т.е. −ELBO)

### Энкодер (CNN₂)

На входе тензор **2×28×28** (batch, channels, height, width):

- Канал 0 — яркость пикселя (нормализованная [0, 1])
- Канал 1 — условие: скаляр лейбла (0…9) повторённый на все пиксели без нормализации

Архитектура:

1. Conv2d(2, **8**, kernel=3, padding=1) + ReLU + MaxPool2d(2×2) → 8×14×14
2. Conv2d(8, **32**, kernel=3, padding=1) + ReLU + MaxPool2d(2×2) → 32×7×7
3. Mean Pooling по пространственным измерениям (avg на 7×7) → вектор размерности 32

Результат: два вектора $\mu$, $\log\sigma^{2}$ размерности **latentDim=32**

### Декодер (CNN₁)

1. Сэмплирование $z = \mu + \varepsilon \cdot \exp(\tfrac{1}{2}\log\sigma^{2})$, $\varepsilon \sim \mathcal{N}(0, I)$ (reparameterization trick), + one-hot лейбл → конкатенация $[z, \mathrm{one\_hot}(\mathrm{label})]$
2. Linear(42 → **49**) + reshape в **1×7×7**
3. ConvTranspose2d(1, 4, kernel=3, stride=2, padding=1, output_padding=1) + ReLU → 4×14×14
4. ConvTranspose2d(4, 1, kernel=3, stride=2, padding=1, output_padding=1) + Sigmoid → **1×28×28**

Декодер выдаёт только $\mu_{out}$ (матожидание выходной картинки) — $\sigma^2 = 1$ фиксировано.

### Инференс

Используется **только декодер**:

- $z \sim \mathcal{N}(0, I)$
- На вход декодеру: $z$ + лейбл
- Выход: сгенерированная картинка 28×28

### Формат ответа API — Base64 PNG в JSON

Сгенерированные картинки отдаются как base64-строка внутри JSON-ответа:

```
POST /generate
Request:  { "label": 3 }
Response: { "image": "data:image/png;base64,iVBOR..." }
```

---

## Структура репозитория

```
client/                     # React + Vite + TS
  src/
    components/             # UI-компоненты
    hooks/                  # Кастомные хуки
    api/                    # Клиент для backend-эндпоинтов
    types/                  # Shared типы (TS)
  index.html
  vite.config.ts
  tsconfig.json

server/                     # FastAPI + PyTorch
  app/
    api/                    # Роуты (FastAPI routers)
    models/                 # CVAE архитектура (PyTorch nn.Module)
    schemas/                # Pydantic-схемы ввода/вывода
    services/               # Логика инференса и обучения
    config.py               # Настройки (pathlib, env)
  main.py                   # Точка входа FastAPI
  train.py                  # Скрипт обучения модели
  requirements.txt

server/weights/              # Чекпоинты модели (.pt / .pth)
  .gitkeep
```

---

## Правила кода

### Frontend

- `strict: true` в `tsconfig.json` — без исключений.
- Только функциональные компоненты + хуки. Классовых компонентов нет.
- `TanStack Query` (`@tanstack/react-query`) для всех запросов к backend. Ручных `useState` + `useEffect` для fetch-логики нет.
- **Tailwind CSS** — утилитарные классы для стилизации. Кастомные CSS-файлы только для Tailwind-директив и переменных.
- **Framer Motion** — анимации генерации, переходы, микро-взаимодействия. Никаких CSS-анимаций вручную.
- Тип `any` запрещён. Использовать `unknown` с последующим type guard, либо конкретный тип.
- Именование файлов: `.tsx` для компонентов, `.ts` для утилит/хуков/типов.

### Backend

- Все входные/выходные данные API — Pydantic-модели (`BaseModel`). Raw dict'ы в роутах запрещены.
- Явная типизация тензоров: `torch.Tensor` аннотации везде, где используется тензор.
- `async def` для роутов, которые делают I/O (даже если PyTorch-инференс синхронный — обернуть через `run_in_executor`).
- Обучение: `train.py` — отдельный CLI-скрипт, не часть API.

### Общий стиль

- **Размер файла ~50 строк**. Если файл превышает ~50 строк — декомпозировать: вынести части в отдельные модули/компоненты/утилиты.
- Чистые функции (no side effects) где возможно.
- Осмысленные имена: `latentVector`, а не `z`; `encodedFeatures`, а не `x_enc`.
- Комментарии — только на русском, только для «почему», никогда для «что».
    ```python
    # Хорда дисперсии > 0 нужна для численной стабильности log
    # НЕ: # Считаем дисперсию
    ```

---

## Команды разработчика

```bash
# Frontend (dev-сервер)
cd client
npm install
npm run dev

# Backend (uvicorn, hot-reload)
cd server
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Обучение модели
cd server
python train.py --epochs 50 --batch-size 128 --lr 1e-3
```

---

## Антипаттерны

| Запрещено                                         | Причина                                            |
| ------------------------------------------------- | -------------------------------------------------- |
| Хардкод путей (`"weights/model.pt"`)              | Использовать `pathlib` + конфиг из env / config.py |
| `any` в TypeScript                                | Подрывает систему типов                            |
| `torch.no_grad()` отсутствует при инференсе       | Лишние вычисления графа, утечка памяти             |
| Блокирующие операции в главном потоке API         | Весь endpoint блокируется                          |
| Генерация без seed'а при дебаге                   | Невоспроизводимость → невозможность отладки        |
| `eval()` / `exec()` / pickle из unsanitized input | RCE-уязвимость                                     |
| Commit файлов `weights/*.pt`                      | В `.gitignore` — модели могут быть сотни МБ        |
| `requests.get()` синхронно в async-роуте          | Блокирует event loop                               |
