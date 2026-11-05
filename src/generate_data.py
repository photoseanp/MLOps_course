"""DVC stage 1: generate a synthetic raw dataset into data/raw_data.csv."""

import random
from pathlib import Path

SEED = 42
NUM_ROWS = 100
OUTPUT = Path("data/raw_data.csv")


def main() -> None:
    random.seed(SEED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("feature1,feature2,target\n")
        for _ in range(NUM_ROWS):
            f1 = round(random.uniform(0, 10), 2)
            f2 = round(random.uniform(0, 10), 2)
            target = int(f1 + f2 > 10)
            f.write(f"{f1},{f2},{target}\n")
    print(f"Raw data generated: {OUTPUT} ({NUM_ROWS} rows).")


if __name__ == "__main__":
    main()
