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

## Быстрый старт

Веса модели в `.gitignore`, поэтому в репозитории их нет: первый запуск состоит из двух этапов —
обучение (один раз) и запуск приложения. На каждом этапе можно выбрать Docker или локальный запуск.

### 1. Обучение модели

Веса сохранятся в `server/weights/cvae.pt`. Во время обучения скачивается датасет `ylecun/mnist`
из Hugging Face — нужен доступ в интернет. Без весов генерация вернёт `503`.

**Docker:**

```bash
docker compose --profile train run --rm trainer
```

**Локально:**

```bash
cd server
pip install -r requirements.txt
python train.py --epochs 20 --batch-size 128 --lr 1e-3
```

Параметры обучения (`--epochs`, `--batch-size`, `--lr`) — у [train.py](server/train.py).

### 2. Запуск приложения

**Docker** (поднимает `server`, `client` и nginx-прокси):

```bash
docker compose up --build
```

Маршрутизация через [nginx.conf](nginx.conf): `http://localhost` — фронтенд,
`http://api.localhost` — backend-API (документация — `/docs`).
`api.localhost` должен резолвиться в `127.0.0.1` — на Windows добавьте в
`C:\Windows\System32\drivers\etc\hosts`:

```
127.0.0.1 api.localhost
```

**Локально** (двумя терминалами):

```bash
# backend — из server/
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# frontend — из client/
npm install && npm run dev
```

Backend — `http://localhost:8000`, фронтенд — `http://localhost:5173`.

CORS-origin'ы и URL API берутся из `.env` (шаблон — [.env.example](.env.example)).

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
CORS-origin'ы и URL API для клиента задаются в `.env` (`CORS_ORIGINS`, `VITE_API_URL`, шаблон — [.env.example](.env.example)); сервер читает их через [config.py](server/app/config.py).

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
compose.yaml                # server + client + nginx-прокси + trainer (обучение)
nginx.conf                  # Роутинг: localhost → фронт, api.localhost → бэк
.env / .env.example         # CORS_ORIGINS, VITE_API_URL (шаблон коммитится)
CLAUDE.md                   # Архитектура модели и правила кода
```

## Архитектура модели

Условный VAE: лейбл $y \in \{0,\dots,9\}$ подаётся и в энкодер (как one-hot каналы), и в декодер
(как one-hot вектор). Латентная размерность $d = $ `LATENT_DIM` $= 32$, дисперсия априори
$\sigma^{2} = $ `PRIOR_VAR` $= 1.0$ ([config.py](server/app/config.py)).

### Распределения

| Компонент               | Формула                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| Априор                  | $p(z) = \mathcal{N}\!\left(0,\ \sigma^{2} I\right)$                                                    |
| Лайклихуд               | $p(x \mid z, y) = \mathrm{Bernoulli}\!\left(x \mid \hat{x}\right)$, $\hat{x} = \mathrm{Decoder}(z, y)$ |
| Вариационный апостериор | $q(z \mid x, y) = \mathcal{N}\!\left(\mu(x, y),\ \mathrm{diag}\, e^{\log\sigma^{2}(x, y)}\right)$      |

Каждый пиксель трактуется как независимая бернуллиевская величина, поэтому декодер через Sigmoid
выдаёт вероятность пикселя $\hat{x} \in [0,1]^{1\times28\times28}$.

### Reparameterization trick

Сэмплирование $z$ сделано дифференцируемым ([cvae.py](server/app/models/cvae.py)):

$$
z = \mu(x, y) + \varepsilon \odot e^{\tfrac{1}{2}\log\sigma^{2}(x, y)}, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

### Функция потерь (ELBO)

$$
\mathcal{L}(x, y) = \underbrace{\mathrm{BCE}(\hat{x}, x)}_{\displaystyle -\log p(x \mid z, y)} \;+\; \beta\, D_{KL}\!\left(q(z \mid x, y)\,\|\, p(z)\right)
$$

KL берётся в замкнутом виде для диагональных гауссиан с неединственным априорном
([losses.py](server/app/services/losses.py)); сумма по латентным осям, среднее по батчу:

$$
D_{KL} = \frac{1}{2} \sum_{j=1}^{d} \left( \ln\sigma^{2} \;-\; \log\sigma_{j}^{2} \;+\; \frac{e^{\log\sigma_{j}^{2}}}{\sigma^{2}} \;+\; \frac{\mu_{j}^{2}}{\sigma^{2}} \;-\; 1 \right)
$$

Здесь $\log\sigma_{j}^{2}$ и $\mu_{j}$ — выходы энкодера (лог-дисперсия и среднее $j$-й латентной
оси), $\sigma^{2}$ — дисперсия априора. При $\sigma^{2} = 1$ лог-член $\ln\sigma^{2}$ обнуляется и
формула сводится к стандартному KL для $\mathcal{N}(0, I)$. $\beta = $ `kl_weight`.

### KL-annealing

Вес $\beta$ — не константа, а задержанное линейное расписание ([train.py:35](server/train.py#L35)):

$$
\beta(e) = \begin{cases} 0, & e \le E/2 \\[4pt] 0.05\,\dfrac{e}{E}, & e > E/2 \end{cases}
$$

где $e$ — номер эпохи, $E$ — всего эпох. Первую половину обучения оптимизируется чистая
реконструкция, затем $\beta$ линейно растёт примерно до $0.05$.

**Зачем это нужно.** Если бы $\beta = 1$ с первой эпохи, оптимизатору было бы выгоднее всего
мгновенно «обнулить» KL-член — столкнуть $q(z \mid x, y)$ к априору независимо от входа $x$. Латент перестаёт нести информацию, энкодер фактически
игнорируется, и модель вырождается в обычный автоэнкодер, где декодер реконструирует картинку
только по лейблу $y$. Это явление — **posterior collapse** (коллапс апостериора). Держа
$\beta \approx 0$ в начале, мы сначала заставляем энкодер и декодер научиться осмысленной
реконструкции и «привязаться» к $z$; KL-давление, прижимающее $z$ к $\mathcal{N}(0,\ \sigma^{2} I)$,
включается лишь позже — к этому моменту модель уже не готова отказаться от латента.

### Кодирование условия

Энкодер получает 11-канальный тензор $x_{\text{in}} \in \mathbb{R}^{B \times 11 \times 28 \times 28}$
([data.py](server/app/services/data.py)):

- канал $0$ — яркость пикселя, нормализованная в $[0,1]$;
- каналы $1\dots10$ — $\mathrm{onehot}(y)$, размноженный на всё пространство $28\times28$.

### Энкодер (CNN₂)

$$
11{\times}28{\times}28 \xrightarrow{\;\text{Conv 3×3 (8), ReLU, MaxPool}\;} 8{\times}14{\times}14 \xrightarrow{\;\text{Conv 3×3 (32), ReLU, MaxPool}\;} 32{\times}7{\times}7 \xrightarrow{\;\text{Flatten}\;} \varphi \in \mathbb{R}^{1568}
$$

Две линейные головы без активации — $\log\sigma^{2}$ должно быть знакопеременной величиной:

$$
\mu = W_{\mu}\,\varphi + b_{\mu}, \qquad \log\sigma^{2} = W_{\sigma^{2}}\,\varphi + b_{\sigma^{2}}, \qquad \mu,\, \log\sigma^{2} \in \mathbb{R}^{d}
$$

### Декодер (CNN₁)

$$
[z;\, \mathrm{onehot}(y)] \in \mathbb{R}^{d+10} \xrightarrow{\;\text{Linear } (d{+}10 \to 256),\, \text{ReLU}\;} h_1 \xrightarrow{\;\text{Linear } (256 \to 784),\, \text{ReLU}\;} h_2 \xrightarrow{\;\text{reshape}\;} 16{\times}7{\times}7
$$

$$
16{\times}7{\times}7 \xrightarrow{\;\text{ConvT } s{=}2 \text{ (32), ReLU}\;} 32{\times}14{\times}14 \xrightarrow{\;\text{ConvT } s{=}2 \text{ (16), ReLU}\;} 16{\times}28{\times}28 \xrightarrow{\;\text{Conv } 3{\times}3 \text{ (1), Sigmoid}\;} \hat{x}
$$

### Инференс (генерация)

При генерации энкодер не нужен — сэмплируем из априора и декодируем conditioned на лейбле:

$$
z \sim \mathcal{N}\!\left(0,\ \sigma^{2} I\right), \qquad \hat{x} = \mathrm{Decoder}(z, y)
$$

Параметры по умолчанию — в [config.py](server/app/config.py): `LATENT_DIM = 32`, `PRIOR_VAR = 1.0`.
