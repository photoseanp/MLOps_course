# MLOps_course

Домашние задания по курсу MLOps: окружение и линтеры, Git, версионирование данных (DVC), тестирование ML-кода (pytest, Hypothesis) и трекинг экспериментов (MLflow).

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

