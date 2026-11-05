"""Training with manual MLflow tracking (params, metrics, tags, artifacts).

Run (with the tracking server started, see README):
    uv run python -m src.train
"""

import io
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.dataset import SyntheticDataset, create_dataloader
from src.models import SimpleClassifier
from src.trainer import Trainer

# MLflow prints emoji; switch Windows console to UTF-8 to avoid UnicodeEncodeError.
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "pytorch_training"
IN_FEATURES = 8
NUM_CLASSES = 2


def run_training(
    lr: float = 0.01,
    epochs: int = 10,
    batch_size: int = 16,
    hidden_dim: int = 32,
    seed: int = 42,
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"run_lr_{lr}_epochs_{epochs}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_type": "SimpleClassifier",
                "framework": "PyTorch",
                "task": "classification",
            }
        )
        mlflow.log_params(
            {
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "seed": seed,
            }
        )

        torch.manual_seed(seed)
        dataset = SyntheticDataset(num_samples=200, in_features=IN_FEATURES, seed=seed)
        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
        model = SimpleClassifier(IN_FEATURES, NUM_CLASSES, hidden_dim=hidden_dim)
        trainer = Trainer(model=model, lr=lr)

        for epoch in range(epochs):
            epoch_loss = trainer.fit(loader, epochs=1)[0]["loss"]
            # step=epoch -> MLflow draws a loss curve over epochs
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"[{run_name}] epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}")

        final = trainer.evaluate(loader)
        mlflow.log_metrics(
            {"final_loss": final["loss"], "train_accuracy": final["accuracy"]}
        )

        # Artifact 1: raw weights file
        model_path = Path("output") / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="weights")

        # Artifact 2: MLflow Model (MLmodel + environment), loadable via mlflow.pytorch
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=dataset.x[:2].numpy(),
            serialization_format="pickle",
        )

        print(f"Run {run_name} finished: accuracy={final['accuracy']:.3f}\n")


if __name__ == "__main__":
    print("--- Run 1: lr=0.05 ---")
    run_training(lr=0.05, epochs=10)

    print("--- Run 2: lr=0.005 ---")
    run_training(lr=0.005, epochs=10)
