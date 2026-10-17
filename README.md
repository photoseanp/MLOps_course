# MLOps_course

Minimal project with [uv](https://docs.astral.sh/uv/), [Ruff](https://docs.astral.sh/ruff/) and [ty](https://docs.astral.sh/ty/).

## Usage

```bash
uv sync                    # create .venv and install dev tools
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
uv run python -c "from mlops_course import add; print(add(2, 3))"
```
