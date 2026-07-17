# CVAE — Генерация рукописных цифр (MNIST)

Full-stack приложение с условным вариационным автокодировщиком (CVAE), обученным на MNIST.
Frontend — интерактивный UI для генерации цифр по лейблу; backend — FastAPI-API для
инференса модели и CLI-скрипт для обучения.

> Подробное описание архитектуры модели и конвенций кода — в [CLAUDE.md](CLAUDE.md).

## Возможности

- 🎨 Генерация изображений цифр 0–9 по выбранному лейблу (условно от лейбла)
- ⚡ FastAPI-инференс: модель загружается один раз при старте, картинки отдаются как base64-PNG
- 🖥️ React + TypeScript UI с анимациями Framer Motion и запросами через TanStack Query
- 🔬 Сам CVAE: CNN-энкодер/декодер, reparameterization trick, BCE-реконструкция + KL с annealing-ом

## Стек технологий

| Слой     | Технологии                                                                         |
| -------- | ---------------------------------------------------------------------------------- |
| Frontend | React 19, Vite 6, TypeScript, Tailwind CSS 4, Framer Motion, TanStack Query, Axios |
| Backend  | FastAPI, PyTorch, Uvicorn                                                          |
| Модель   | Conditional VAE (CNN), MNIST                                                       |
| Данные   | `ylecun/mnist` из Hugging Face (`datasets`)                                        |

## Архитектура модели

Условный VAE: лейбл подаётся и в энкодер (как one-hot каналы), и в декодер (как one-hot вектор).

**Энкодер** — вход `11×28×28` (канал 0 — яркость пикселя, каналы 1–10 — one-hot лейбла,
размноженный на всё пространство):

`Conv2d(11→8, k3, p1)+ReLU+MaxPool` → 8×14×14 → `Conv2d(8→32, k3, p1)+ReLU+MaxPool` → 32×7×7 →
`Flatten` (1568) → две головы `Linear(1568→latent_dim)`: `mu` и `log_var`.

**Reparameterization** (`CVAE.forward`): `z = mu + ε·exp(½·log_var)`, `ε ~ N(0, I)`.

**Декодер** — `concat[z, one_hot(label)]` (`latent_dim + 10`) → `Linear(→256)+ReLU` →
`Linear(→784)+ReLU` → reshape `16×7×7` → `ConvT(16→32, s2)+ReLU` → 32×14×14 →
`ConvT(32→16, s2)+ReLU` → 16×28×28 → `Conv2d(16→1, k3, p1)+Sigmoid` → `1×28×28`.

**Инференс** использует только декодер: `z ~ N(0, prior_var·I)` + лейбл → вероятность пикселя.

**Loss**: `BCE(x̂, x) + β·KL`, где `β = kl_weight` (KL-annealing — см. [train.py:35](server/train.py#L35)).
KL считается в общем виде для априори `N(0, prior_var·I)` (лог-член нужен при `prior_var ≠ 1`),
суммируется по латентным осям и усредняется по батчу ([losses.py](server/app/services/losses.py)).

Параметры по умолчанию заданы в [config.py](server/app/config.py): `LATENT_DIM=32`, `PRIOR_VAR=1.0`.

## Структура репозитория

```
client/                     # React + Vite + TS (UI генерации)
  src/
    components/             # UI-компоненты
    hooks/                  # Кастомные хуки
    api/                    # Клиент для backend-эндпоинтов
    types/                  # Shared типы (TS)
server/                     # FastAPI + PyTorch
  app/
    api/                    # Роуты (FastAPI routers)
    models/                 # CVAE: encoder.py, decoder.py, cvae.py
    schemas/                # Pydantic-схемы ввода/вывода
    services/               # Инференс, загрузка данных, лоссы
    config.py               # Пути, латентная размерность, prior_var, хост/порт
  main.py                   # Точка входа FastAPI
  train.py                  # CLI-скрипт обучения модели
  requirements.txt
  weights/                  # Чекпоинты модели (.pt) — в .gitignore
CLAUDE.md                   # Архитектура модели и правила кода
```

## Быстрый старт

### 1. Обучение модели

```bash
cd server
python train.py --epochs 50 --batch-size 128 --lr 1e-3
```

Веса сохранятся в `server/weights/cvae.pt`. Во время обучения скачивается датасет
`ylecun/mnist` из Hugging Face — нужен доступ в интернет. График обучения пишется в лог
(`loss`, `recon`, `kl`, `lr`, `kl_weight` за эпоху). Без обученных весов генерация вернёт `503`.

### 2. Backend (API + инференс)

```bash
cd server
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API поднимется на `http://localhost:8000`, документация — на `/docs`.

### 3. Frontend (UI)

```bash
cd client
npm install
npm run dev
```

Dev-сервер Vite — на `http://localhost:5173`. CORS в [main.py](server/main.py) уже разрешает
этот origin. Перед запуском убедитесь, что backend поднят на `:8000`.

## API

### `POST /generate`

Генерирует изображение заданной цифры из априорного распределения `p(z)`.

```jsonc
// Request
{ "label": 3 }

// Response (200)
{ "image": "data:image/png;base64,iVBORw0KGgo..." }
```

`label` — целое `0–9`. Возвращает `503`, если модель не загружена (веса отсутствуют).

## Конфигурация

Основные параметры — в [server/app/config.py](server/app/config.py):

| Параметр        | Значение по умолчанию | Описание                                         |
| --------------- | --------------------- | ------------------------------------------------ |
| `LATENT_DIM`    | `32`                  | Размер латентного пространства                   |
| `PRIOR_VAR`     | `1.0`                 | Дисперсия априорного `p(z) = N(0, PRIOR_VAR·I)`  |
| `MODEL_PATH`    | `weights/cvae.pt`     | Путь к чекпоинту                                 |
| `HOST` / `PORT` | `0.0.0.0` / `8000`    | Адрес uvicorn (переопределяется CLI-аргументами) |

Параметры обучения (`--epochs`, `--batch-size`, `--lr`) — у [train.py](server/train.py).
CORS-origin фронтенда (`http://localhost:5173`) задаётся в [server/main.py](server/main.py).
