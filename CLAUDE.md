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

- **Приор**: $p(z) = \mathcal{N}(0, \sigma^{2} I)$ — дисперсия $\sigma^{2} =$ `PRIOR_VAR` ([config.py](server/app/config.py)), единая для всех латентных размерностей (диагональная ковариация); `PRIOR_VAR = 1.0` сводит к стандартному $\mathcal{N}(0, I)$
- **Лайклихуд**: $p(x \mid z) = \mathrm{Bernoulli}(\mathrm{Decoder}(z))$ — каждый пиксель трактуется как независимая бернуллиевская величина, декодер через Sigmoid выдаёт вероятность
- **Апостериорный аналог**: $q(z \mid x) = \mathcal{N}(\mu(x), \mathrm{diag}(\exp(\log\sigma^{2}(x))))$
- **Loss (как в [losses.py](server/app/services/losses.py))**: $\mathrm{BCE}(\hat{x}, x)_{\mathrm{mean}} + \beta \cdot D_{KL}(q(z|x) \| p(z))$, где $\beta =$ `kl_weight`. BCE соответствует $-\log p(x|z)$ для бернуллиевского лайклихуда; KL берётся суммой по латентным размерностям и средним по батчу. `kl_weight = 1.0` (без annealing, см. [train.py:35](server/train.py#L35))

### Энкодер (CNN₂)

На входе тензор **11×28×28** (batch, channels, height, width):

- Канал 0 — яркость пикселя (нормализованная [0, 1])
- Каналы 1–10 — one-hot лейбла: 10 бинарных каналов, каждый повторён на всё пространство 28×28

Условие собирает `prepare_condition` в [services/data.py](server/app/services/data.py).

Архитектура:

1. Conv2d(11, **8**, kernel=3, padding=1) + ReLU + MaxPool2d(2×2) → 8×14×14
2. Conv2d(8, **32**, kernel=3, padding=1) + ReLU + MaxPool2d(2×2) → 32×7×7
3. Flatten → вектор размерности 32·7·7 = **1568**

Результат: две `Linear`-головы (`mu_head`, `log_var_head`: 1568 → latent_dim) дают $\mu$ и $\log\sigma^{2}$ размерности **latent_dim = 128** (`LATENT_DIM` в [config.py](server/app/config.py)).

### Декодер (CNN₁)

Reparameterization trick живёт в [CVAE.forward](server/app/models/cvae.py#L20-L23): $z = \mu + \varepsilon \cdot \exp(\tfrac{1}{2}\log\sigma^{2})$, $\varepsilon \sim \mathcal{N}(0, I)$. Декодер получает уже готовый $z$.

1. Конкатенация $[z, \mathrm{one\_hot}(\mathrm{label})]$ → вектор размерности **latent_dim + 10 = 138**
2. Проекция `Linear(138 → 256)` + ReLU + `Linear(256 → 784)` + ReLU + reshape в **16×7×7**
3. ConvTranspose2d(16, 32, kernel=3, stride=2, padding=1, output_padding=1) + ReLU → 32×14×14
4. ConvTranspose2d(32, 16, kernel=3, stride=2, padding=1, output_padding=1) + ReLU → 16×28×28
5. Conv2d(16, 1, kernel=3, padding=1) + Sigmoid → **1×28×28**

Декодер выдаёт вероятность пикселя $\hat{x}$ (бернуллиевский параметр), что вместе с BCE-реконструкцией эквивалентно $-\log p(x|z)$.

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
