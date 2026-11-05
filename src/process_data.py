"""DVC stage 2: read data/raw_data.csv and write metrics to output/metrics.json."""

import csv
import json
from pathlib import Path

INPUT = Path("data/raw_data.csv")
OUTPUT = Path("output/metrics.json")


def main() -> None:
    with INPUT.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sample_count = len(rows)
    positive = sum(int(row["target"]) for row in rows)
    positive_rate = positive / sample_count if sample_count else 0.0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"sample_count": sample_count, "positive_rate": round(positive_rate, 4)}
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(f"Metrics saved: {sample_count} samples, positive_rate={positive_rate:.4f}.")


if __name__ == "__main__":
    main()
