# MLOps_course

Домашние задания по курсу MLOps: окружение и линтеры, Git, версионирование данных (DVC), тестирование ML-кода (pytest, Hypothesis) и трекинг экспериментов (MLflow).

---

## Структура проекта

```text
MLOps_course/
├── .dvc/                  # конфигурация DVC (создаётся командой dvc init)
├── .github/
│   └── ci.yml             # CI: uv sync, ruff, ty, pytest
├── data/                  # данные, версионируются DVC (в .gitignore)
│   └── raw_data.csv
├── output/
│   └── metrics.json       # метрики DVC-пайплайна
├── src/
│   ├── __init__.py        # функция add (ДЗ 1)
│   ├── generate_data.py   # стадия DVC: генерация данных (ДЗ 3)
│   ├── process_data.py    # стадия DVC: подсчёт метрик (ДЗ 3)
│   ├── models.py          # LinearBlock, SimpleClassifier (ДЗ 4)
│   ├── dataset.py         # SyntheticDataset, create_dataloader (ДЗ 4)
│   ├── trainer.py         # Trainer: fit, evaluate, smoke, overfit (ДЗ 4)
│   ├── train.py           # обучение с ручным трекингом MLflow (ДЗ 5)
│   └── train_autolog.py   # Lightning + mlflow.pytorch.autolog (ДЗ 5)
├── tests/
│   ├── conftest.py
│   ├── test_main.py
│   ├── test_block.py
│   ├── test_model.py
│   ├── test_dataloader.py
│   ├── test_train.py
│   └── test_e2e.py
├── .dvcignore
├── .gitignore
├── .python-version        # Python 3.12
├── dvc.yaml               # стадии DVC-пайплайна
├── dvc.lock               # хэши артефактов пайплайна
├── pyproject.toml         # зависимости uv и настройки Ruff / ty / pytest
├── uv.lock                # зафиксированные версии зависимостей
└── README.md
```

---

## Быстрый старт

Проект управляется пакетным менеджером [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/photoseanp/MLOps_course.git
cd MLOps_course
uv sync
```

---

## ДЗ 1. uv, Ruff и ty

Окружение создаётся через `uv`, зависимости зафиксированы в `uv.lock`. Ruff (линтер и форматтер) и ty (проверка типов) подключены как dev-зависимости.

Настройки в `pyproject.toml`:

- `tool.ruff`: длина строки 88 символов, правила `E`, `F`, `I`, `UP`, `B`.
- `tool.ty.environment`: Python 3.12.
- `tool.ty.rules`: строгий уровень `error` для ключевых проверок типов.

```bash
uv run ruff check .          # линтер
uv run ruff format --check . # форматирование
uv run ty check              # типы
```

---

## ДЗ 2. Работа с Git

Отработан Git-flow на учебном скрипте `src/train.py`:

- ветки `main` (стабильная), `develop` (интеграция) и feature-ветки `feature/*`;
- `feature/baseline-train`: базовый скрипт обучения с фиксированным seed;
- `feature/experiment-lr-schedule`: эксперимент с cosine LR schedule (`lr=1e-2`, `epochs=10`);
- `feature/experiment-high-lr`: альтернативный эксперимент с `lr=5e-2`, при слиянии получен и вручную разрешён конфликт (merge conflict) в `src/train.py`;
- ML-артефакты (`*.pt`, `*.ckpt`, `data/`, `mlruns/` и т. д.) добавлены в `.gitignore`;
- коммиты оформлены по [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `experiment:`, `chore:`, `test:`, `docs:`;
- изменения попадают в `main` через Pull Request из `develop`.

---

## ДЗ 3. Версионирование данных и пайплайн (DVC)

Remote-хранилище — локальная папка вне репозитория:

```bash
uv run dvc init
uv run dvc remote add -d mylocalremote ~/.dvc_local_storage
```

Пайплайн `dvc.yaml` состоит из двух стадий:

1. `generate_data` — `uv run python src/generate_data.py` создаёт `data/raw_data.csv` (100 строк, seed 42). Файл версионируется DVC, а не Git.
2. `process_data` — `uv run python src/process_data.py` читает `data/raw_data.csv` и пишет метрики в `output/metrics.json` (`sample_count`, `positive_rate`).

```bash
uv run dvc repro         # запуск пайплайна
uv run dvc metrics show  # метрики
uv run dvc push          # отправить данные в remote
uv run dvc pull          # скачать данные из remote
```

---

## ДЗ 4. Тестирование ML-кода (TDD, pytest, Hypothesis)

Код написан по TDD: сначала коммит с падающими тестами (red), затем реализация (green).

Компоненты (`src/`):

- `models.py`: `LinearBlock` (Linear + ReLU) и `SimpleClassifier`.
- `dataset.py`: воспроизводимый `SyntheticDataset` и `create_dataloader`.
- `trainer.py`: `Trainer` с `fit` (в т. ч. `fast_dev_run`), `evaluate`, `train_epoch_with_checks` (smoke) и `overfit_batch`.

Пирамида тестов (метки pytest):

- `unit` — размерности выходов, ReLU, длина датасета, сборка батчей.
- `property` — инварианты формы на случайных размерах (Hypothesis).
- `smoke` — dry run на одном батче; loss конечен и убывает, у всех параметров `grad is not None`.
- `overfit` — переобучение на одном батче до loss < 0.05.
- `e2e` — полный цикл: данные → обучение → оценка (accuracy > 0.8) → инференс.

```bash
uv run pytest -v --cov=src --cov-report=term-missing
uv run pytest -m smoke -v
uv run pytest -m property -v
```

---

## ДЗ 5. Трекинг экспериментов (MLflow)

Хранилище MLflow:

- Backend Store: SQLite `mlflow.db` (параметры, метрики, теги);
- Artifact Store: папка `./mlruns` (веса и MLflow Model);
- Tracking Server: `http://127.0.0.1:5000`.

Скрипты:

- `src/train.py` — эксперимент `pytorch_training`, ручное логирование: параметры (`learning_rate`, `epochs`, `batch_size`, `hidden_dim`, `seed`), метрика `train_loss` по эпохам (`step=epoch`), `final_loss` и `train_accuracy`, теги (`model_type`, `framework`, `task`), артефакты `weights/model.pt` и MLflow Model. Два прогона: `lr=0.05` и `lr=0.005`.
- `src/train_autolog.py` — эксперимент `pytorch_autolog_experiment`, PyTorch Lightning + `mlflow.pytorch.autolog()`: параметры, метрики и модель логируются без ручных вызовов `log_*`. Два прогона: `lr=0.02` и `lr=0.002`.

```bash
# терминал 1: сервер
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# терминал 2: обучение
uv run python -m src.train
uv run python -m src.train_autolog
```

В UI (`http://127.0.0.1:5000`) выберите прогоны эксперимента и нажмите **Compare**: видны различия гиперпараметров и совмещённые кривые `train_loss`.
