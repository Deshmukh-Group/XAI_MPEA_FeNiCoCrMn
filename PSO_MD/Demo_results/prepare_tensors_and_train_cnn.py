#!/usr/bin/env python3
"""Reproduce the MD-to-tensor workflow on Demo_results birds 0, 1, and 2.

Examples
--------
Check that the raw MD structures reproduce the checked-in processed files:

    python3 prepare_tensors_and_train_cnn.py --check-processed --prepare-only

Rebuild the processed text files and create deterministic tensor splits:

    python3 prepare_tensors_and_train_cnn.py --rebuild-processed --prepare-only

Train the elastic-property demo model after preparing the tensors:

    python3 prepare_tensors_and_train_cnn.py --train --epochs 100

Use ``--property usfe`` to prepare or train the USFE dataset/model instead.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_BIRDS = (0, 1, 2)
DEFAULT_SAMPLES_PER_REPLICA = 49  # epochs 0-48 in the published processed files
REPLICAS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild processed PSO/MD data, create tensors, and optionally train the CNN."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing numeric bird folders (default: script directory)",
    )
    parser.add_argument(
        "--birds",
        type=int,
        nargs="+",
        default=list(DEFAULT_BIRDS),
        help="bird folders to use (default: 0 1 2)",
    )
    parser.add_argument(
        "--samples-per-replica",
        type=int,
        default=DEFAULT_SAMPLES_PER_REPLICA,
        help="number of sequential epochs used for each replica (default: 49)",
    )
    process_group = parser.add_mutually_exclusive_group()
    process_group.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="regenerate type_file*.txt and prop*.txt from saved MD files",
    )
    process_group.add_argument(
        "--check-processed",
        action="store_true",
        help="verify that raw MD files reproduce the existing processed files",
    )
    parser.add_argument(
        "--property",
        choices=("elastic", "usfe"),
        default="elastic",
        help="dataset/model to prepare (default: elastic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="tensor/model output directory (default: generated/<property>)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="keep the highest values of the first target after sorting; 0 keeps all",
    )
    parser.add_argument("--train", action="store_true", help="train after preparing tensors")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="explicitly stop after preparing tensors (the default unless --train is supplied)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    return parser.parse_args()


def nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_lammps_atom_types(path: Path) -> list[int]:
    """Return atom types from a LAMMPS data file, ordered by atom ID."""
    lines = path.read_text().splitlines()
    atom_count = None
    atoms_start = None

    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 2 and fields[1] == "atoms":
            atom_count = int(fields[0])
        if line.strip().startswith("Atoms"):
            atoms_start = index + 1
            break

    if atom_count is None or atoms_start is None:
        raise ValueError(f"Could not locate the atom count or Atoms section in {path}")

    records: list[tuple[int, int]] = []
    for line in lines[atoms_start:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            atom_id, atom_type = int(fields[0]), int(fields[1])
        except ValueError:
            if records:
                break
            continue
        records.append((atom_id, atom_type))
        if len(records) == atom_count:
            break

    if len(records) != atom_count:
        raise ValueError(f"Expected {atom_count} atoms in {path}, found {len(records)}")
    records.sort(key=lambda record: record[0])
    if [atom_id for atom_id, _ in records] != list(range(1, atom_count + 1)):
        raise ValueError(f"Atom IDs are not consecutive in {path}")

    atom_types = [atom_type for _, atom_type in records]
    if not all(1 <= atom_type <= 5 for atom_type in atom_types):
        raise ValueError(f"Unexpected atom type in {path}; expected integer values 1-5")
    return atom_types


def create_processed_content(bird_dir: Path, samples_per_replica: int) -> dict[str, str]:
    """Create the four processed text datasets for one bird in replica-major order."""
    elastic_labels = {
        replica: nonempty_lines(bird_dir / f"elastic{replica}.txt")
        for replica in REPLICAS
    }
    gsfe_rows = [line.split() for line in nonempty_lines(bird_dir / "gsfe.txt")]

    for replica, labels in elastic_labels.items():
        if len(labels) < samples_per_replica:
            raise ValueError(
                f"{bird_dir}/elastic{replica}.txt contains {len(labels)} rows; "
                f"{samples_per_replica} are required"
            )
    if len(gsfe_rows) < samples_per_replica:
        raise ValueError(
            f"{bird_dir}/gsfe.txt contains {len(gsfe_rows)} rows; "
            f"{samples_per_replica} are required"
        )
    if any(len(row) != len(REPLICAS) for row in gsfe_rows[:samples_per_replica]):
        raise ValueError(f"Each selected row of {bird_dir}/gsfe.txt must contain three values")

    generated: dict[str, list[str]] = {
        "type_file.txt": [],
        "prop.txt": [],
        "type_file_usf.txt": [],
        "prop_usf.txt": [],
    }
    for replica in REPLICAS:
        for epoch in range(samples_per_replica):
            elastic_structure = (
                bird_dir / "file_save" / f"data_100_use{replica}_{epoch}.dat"
            )
            usfe_structure = (
                bird_dir / "file_save111" / f"data_111_use{replica}_{epoch}.dat"
            )
            generated["type_file.txt"].append(
                repr(read_lammps_atom_types(elastic_structure))
            )
            generated["prop.txt"].append(
                " ".join(elastic_labels[replica][epoch].split())
            )
            generated["type_file_usf.txt"].append(
                repr(read_lammps_atom_types(usfe_structure))
            )
            generated["prop_usf.txt"].append(
                str(float(gsfe_rows[epoch][replica - 1]))
            )

    return {name: "\n".join(rows) + "\n" for name, rows in generated.items()}


def rebuild_or_check_processed(
    bird_dirs: list[Path], samples_per_replica: int, check_only: bool
) -> None:
    action = "Checking" if check_only else "Rebuilding"
    for bird_dir in bird_dirs:
        print(f"{action} processed files for bird {bird_dir.name}")
        generated = create_processed_content(bird_dir, samples_per_replica)
        for name, content in generated.items():
            path = bird_dir / name
            if check_only:
                if not path.is_file() or path.read_text() != content:
                    raise ValueError(f"Raw-to-processed validation failed: {path}")
            else:
                path.write_text(content)
    print(f"{action} complete for {len(bird_dirs)} birds")


def read_type_file(path: Path) -> list[list[int]]:
    records = []
    for line_number, line in enumerate(nonempty_lines(path), start=1):
        value = ast.literal_eval(line)
        if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
            raise ValueError(f"Invalid atom-type record at {path}:{line_number}")
        records.append(value)
    return records


def read_property_file(path: Path) -> list[list[float]]:
    return [[float(value) for value in line.split()] for line in nonempty_lines(path)]


def load_dataset(
    bird_dirs: list[Path], property_name: str, top: int
) -> tuple[list[list[int]], list[list[float]], dict[str, int]]:
    if property_name == "elastic":
        type_name, property_file_name = "type_file.txt", "prop.txt"
        expected_atoms, expected_targets = 4000, 5
    else:
        type_name, property_file_name = "type_file_usf.txt", "prop_usf.txt"
        expected_atoms, expected_targets = 3600, 1

    combined: list[tuple[list[int], list[float]]] = []
    per_bird: dict[str, int] = {}
    for bird_dir in bird_dirs:
        structures = read_type_file(bird_dir / type_name)
        properties = read_property_file(bird_dir / property_file_name)
        if len(structures) != len(properties):
            raise ValueError(
                f"Record mismatch for bird {bird_dir.name}: "
                f"{len(structures)} structures versus {len(properties)} properties"
            )
        for record in structures:
            if len(record) != expected_atoms:
                raise ValueError(
                    f"Bird {bird_dir.name} has a {len(record)}-atom record; "
                    f"{expected_atoms} expected for {property_name}"
                )
        for record in properties:
            if len(record) != expected_targets:
                raise ValueError(
                    f"Bird {bird_dir.name} has {len(record)} targets; "
                    f"{expected_targets} expected for {property_name}"
                )
        combined.extend(zip(structures, properties))
        per_bird[bird_dir.name] = len(structures)

    # Remove the error sentinel emitted when the MD calculation fails.
    clean = []
    for structure, properties in combined:
        failed = (
            properties[0] == 1000.0
            and (len(properties) == 1 or properties[1] == 1000.0)
        )
        if not failed:
            clean.append((structure, properties))

    clean.sort(key=lambda item: item[1][0], reverse=True)
    if top > 0:
        clean = clean[:top]
    if len(clean) < 10:
        raise ValueError(f"Only {len(clean)} valid records remain; at least 10 are required")

    structures = [item[0] for item in clean]
    properties = [item[1] for item in clean]
    return structures, properties, per_bird


def prepare_splits(
    structures: list[list[int]],
    properties: list[list[float]],
    output_dir: Path,
    seed: int,
    property_name: str,
    per_bird: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_all = torch.tensor(structures, dtype=torch.long)
    y_all = torch.tensor(properties, dtype=torch.float32)

    # First hold out 20%; then use 10% of the remaining 80% for validation.
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.20, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=seed + 1
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "X_train.pt": x_train,
        "X_val.pt": x_val,
        "X_test.pt": x_test,
        "y_train.pt": y_train,
        "y_val.pt": y_val,
        "y_test.pt": y_test,
    }
    for name, tensor in tensors.items():
        torch.save(tensor, output_dir / name)

    metadata = {
        "property": property_name,
        "birds": per_bird,
        "seed_stage_1": seed,
        "seed_stage_2": seed + 1,
        "split_method": "random structure-level split",
        "fractions": {"train": 0.72, "validation": 0.08, "test": 0.20},
        "samples": {
            "all": len(x_all),
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
        },
        "input_length": x_all.shape[1],
        "number_of_targets": y_all.shape[1],
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Prepared {len(x_all)} {property_name} samples: "
        f"{len(x_train)} train, {len(x_val)} validation, {len(x_test)} test"
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


class SharedConvCNN(nn.Module):
    """Original shared-convolution 1D-CNN, generalized to both input lengths."""

    def __init__(self, input_length: int, number_of_targets: int):
        super().__init__()
        self.embedding = nn.Embedding(6, 8)
        self.convolution = nn.Conv1d(8, 8, kernel_size=3)
        self.pooling = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros((1, input_length), dtype=torch.long)
            flattened_size = self.features(dummy).shape[1]

        self.linear1 = nn.Linear(flattened_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, number_of_targets)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).permute(0, 2, 1)
        for _ in range(5):
            x = self.relu(self.pooling(self.convolution(x)))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return self.linear5(x)


class EarlyStopping:
    def __init__(self, patience: int, checkpoint: Path):
        self.patience = patience
        self.checkpoint = checkpoint
        self.best_loss = float("inf")
        self.counter = 0

    def update(self, validation_loss: float, model: nn.Module) -> bool:
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
        return self.counter >= self.patience


def mean_epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train_model(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    x_train, x_val, x_test, y_train, y_val, y_test = tensors
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    model = SharedConvCNN(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    checkpoint = output_dir / "checkpoint.pt"
    early_stopping = EarlyStopping(patience, checkpoint)
    train_losses, validation_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = mean_epoch_loss(model, train_loader, criterion, device, optimizer)
        validation_loss = mean_epoch_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"[{epoch:>{len(str(epochs))}}/{epochs}] "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )
        if early_stopping.update(validation_loss, model):
            print(f"Early stopping after {epoch} epochs")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions, observations = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions.append(model(x_batch.to(device)).cpu())
            observations.append(y_batch)
    prediction = torch.cat(predictions)
    observation = torch.cat(observations)
    mse = torch.mean((prediction - observation) ** 2, dim=0)
    denominator = torch.sum((observation - observation.mean(dim=0)) ** 2, dim=0)
    r2 = 1.0 - torch.sum((prediction - observation) ** 2, dim=0) / denominator
    metrics = {"test_mse": mse.tolist(), "test_r2": r2.tolist()}
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    (output_dir / "train_loss.txt").write_text(
        "\n".join(str(value) for value in train_losses) + "\n"
    )
    (output_dir / "validation_loss.txt").write_text(
        "\n".join(str(value) for value in validation_losses) + "\n"
    )
    figure = plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training loss")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output_dir / "loss_plot.png", dpi=150)
    plt.close(figure)
    print(f"Test metrics: {metrics}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    bird_dirs = [data_dir / str(bird) for bird in args.birds]
    missing = [path for path in bird_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing requested bird directories: {missing}")

    if args.rebuild_processed or args.check_processed:
        rebuild_or_check_processed(
            bird_dirs,
            samples_per_replica=args.samples_per_replica,
            check_only=args.check_processed,
        )

    structures, properties, per_bird = load_dataset(
        bird_dirs, args.property, args.top
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir / "generated" / args.property
    )
    tensors = prepare_splits(
        structures, properties, output_dir, args.seed, args.property, per_bird
    )

    if args.train and not args.prepare_only:
        train_model(
            tensors,
            output_dir,
            args.seed,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
        )
    else:
        print("Tensor preparation complete; use --train to train the CNN")


if __name__ == "__main__":
    main()
