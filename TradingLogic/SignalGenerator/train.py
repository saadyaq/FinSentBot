import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import matplotlib.pyplot as plt
import seaborn as sns

from .models import LSTMSignalGenerator, SimpleMLP


class TradingDataset(Dataset):
    """Dataset PyTorch pour les données de trading."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_length: int = 20,
        use_sequences: bool = True,
    ):
        """
        Args:
            dataframe: DataFrame contenant les features et la colonne 'action'
            sequence_length: Longueur des séquences pour le modèle LSTM
            use_sequences: Génère ou non des séquences temporelles
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("Le dataframe fourni pour l'entraînement est vide.")

        df = dataframe.copy()
        df = df.dropna(subset=["action"])
        if df.empty:
            raise ValueError("Aucune action valide trouvée dans le dataset.")

        self.sequence_length = max(1, int(sequence_length))
        self.use_sequences = bool(use_sequences)

        self.feature_cols = [
            "sentiment_score",
            "price_now",
            "price_future",
            "variation",
        ]

        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Colonnes de features manquantes dans le dataset: {missing_features}"
            )

        df["symbol"] = df.get("symbol", "__GLOBAL__").fillna("__GLOBAL__")
        if "news_timestamp" in df.columns:
            df["news_timestamp"] = pd.to_datetime(
                df["news_timestamp"], errors="coerce"
            )
        else:
            df["news_timestamp"] = pd.NaT

        df = df.sort_values(
            ["symbol", "news_timestamp"], kind="mergesort"
        ).reset_index(drop=True)

        # Préparation des features numériques
        features = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        features = features.ffill().bfill().fillna(0.0)

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features.values).astype(np.float32)

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(df["action"].astype(str)).astype(
            np.int64
        )

        symbols = df["symbol"].astype(str).tolist()

        timestamps = df["news_timestamp"].tolist()

        if self.use_sequences:
            self.X, self.y, self.sample_info = self._build_sequences(
                scaled_features, labels, symbols, timestamps, self.sequence_length
            )
        else:
            self.X = scaled_features
            self.y = labels
            self.sample_info = list(zip(symbols, timestamps))

        print(f"Dataset créé: {len(self.X)} échantillons ({'seq' if self.use_sequences else 'flat'})")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Distribution: {np.bincount(self.y)}")

    @staticmethod
    def _build_sequences(
        features: np.ndarray,
        labels: np.ndarray,
        symbols: list[str],
        timestamps: list[pd.Timestamp],
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray, list[Tuple[str, pd.Timestamp]]]:
        """Construit des séquences temporelles par symbole."""
        symbol_to_indices: Dict[str, list[int]] = {}
        for idx, symbol in enumerate(symbols):
            symbol_to_indices.setdefault(symbol, []).append(idx)

        sequences = []
        seq_labels = []
        feature_dim = features.shape[1]
        sample_info: list[Tuple[str, pd.Timestamp]] = []

        for symbol, indices in symbol_to_indices.items():
            for pos, current_idx in enumerate(indices):
                start_pos = max(0, pos - sequence_length + 1)
                window_indices = indices[start_pos : pos + 1]
                window = features[window_indices]

                if len(window) < sequence_length:
                    pad_len = sequence_length - len(window)
                    padding = np.zeros((pad_len, feature_dim), dtype=np.float32)
                    window = np.vstack((padding, window))

                sequences.append(window.astype(np.float32))
                seq_labels.append(labels[current_idx])
                sample_info.append(
                    (symbol, timestamps[current_idx] if timestamps else pd.NaT)
                )

        return np.stack(sequences), np.array(seq_labels, dtype=np.int64), sample_info

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        features = self.X[idx]
        label = int(self.y[idx])

        if self.use_sequences:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
            )

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class ModelTrainer:
    """Gestionnaire d'entraînement des modèles de signaux."""

    def __init__(self, model_type: str = "lstm", device: str = "auto"):
        """
        Args:
            model_type: "lstm" ou "mlp"
            device: "cpu", "cuda" ou "auto"
        """
        self.model_type = model_type.lower()
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else device
            if device != "auto"
            else "cpu"
        )

        print(f"Entraînement configuré : {self.model_type.upper()} sur {self.device}")

        self.model: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_cols: list[str] | None = None
        self.sequence_length: int = 1
        self.sample_info: list[Tuple[str, pd.Timestamp]] | None = None
        self.best_val_loss: float = float("inf")
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def prepare_data(
        self,
        csv_path: str,
        test_size: float = 0.2,
        batch_size: int = 32,
        sequence_length: int = 20,
        dataloader_timeout: float = 0.0,
        num_workers: int = 0,
        split_strategy: str = "random",
        val_symbols: list[str] | None = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prépare les DataLoaders d'entraînement et validation."""
        df = pd.read_csv(csv_path)
        print(f"Données chargées: {len(df)} échantillons depuis {csv_path}")

        use_sequences = self.model_type == "lstm"
        dataset = TradingDataset(
            df, sequence_length=sequence_length, use_sequences=use_sequences
        )

        self.sample_info = dataset.sample_info

        split_strategy = (split_strategy or "random").lower()
        if split_strategy not in {"random", "time", "symbol"}:
            raise ValueError(f"Split strategy inconnue: {split_strategy}")

        if split_strategy == "random":
            val_size = max(1, int(len(dataset) * test_size))
            train_size = len(dataset) - val_size
            if train_size <= 0:
                raise ValueError("Dataset trop petit pour créer un split train/validation.")

            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )
        else:
            indices = np.arange(len(dataset))

            if split_strategy == "time":
                val_size = max(1, int(len(dataset) * test_size))
                timestamps = []
                for _, ts in self.sample_info:
                    if isinstance(ts, pd.Timestamp):
                        timestamp = ts
                    else:
                        timestamp = pd.to_datetime(ts) if ts else pd.NaT
                    if pd.isna(timestamp):
                        timestamp = pd.Timestamp.min
                    timestamps.append(timestamp.value)

                timestamps = np.array(timestamps, dtype=np.int64)
                sorted_idx = np.argsort(timestamps)
                val_indices = sorted_idx[-val_size:].tolist()
                train_indices = sorted_idx[:-val_size].tolist()
                if not train_indices or not val_indices:
                    raise ValueError("Split temporel invalide: train ou validation vide.")
            else:  # symbol-based
                if not val_symbols:
                    raise ValueError(
                        "La stratégie 'symbol' nécessite de fournir --val-symbols."
                    )
                val_set = {s.strip() for s in val_symbols if s.strip()}
                if not val_set:
                    raise ValueError("Aucun symbole valide fourni pour --val-symbols.")

                val_indices = [
                    idx for idx, (symbol, _) in enumerate(self.sample_info) if symbol in val_set
                ]
                train_indices = [
                    idx for idx, (symbol, _) in enumerate(self.sample_info) if symbol not in val_set
                ]
                if not val_indices:
                    raise ValueError(
                        "Aucun échantillon trouvé pour les symboles de validation fournis."
                    )
                if not train_indices:
                    raise ValueError("Tous les échantillons appartiennent aux symboles de validation.")

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

        timeout_value = dataloader_timeout if (dataloader_timeout > 0 and num_workers > 0) else 0.0
        if dataloader_timeout > 0 and num_workers == 0:
            print(
                "⚠️ Timeout DataLoader ignoré car num_workers=0 (limitation PyTorch). "
                "Augmentez --num-workers pour activer le timeout."
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            timeout=timeout_value,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            timeout=timeout_value,
            num_workers=num_workers,
        )

        input_dim = dataset.X.shape[-1]
        if self.model_type == "lstm":
            self.model = LSTMSignalGenerator(input_dim=input_dim)
        else:
            self.model = SimpleMLP(input_dim=input_dim)

        self.model.to(self.device)

        self.scaler = dataset.scaler
        self.label_encoder = dataset.label_encoder
        self.feature_cols = dataset.feature_cols
        self.sequence_length = dataset.sequence_length

        print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
        return train_loader, val_loader

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer | None,
        criterion: nn.Module,
        training: bool = True,
        grad_clip: float | None = None,
    ) -> Tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Le modèle n'a pas été initialisé. Appelez prepare_data d'abord.")

        epoch_loss = 0.0
        correct = 0
        total = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                logits, _ = self.model(features)
                loss = criterion(logits, labels)

                if training and optimizer is not None:
                    loss.backward()
                    if grad_clip is not None:
                        clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()

                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += batch_size

        avg_loss = epoch_loss / max(1, total)
        accuracy = correct / max(1, total)
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ) -> Dict[str, list[float]]:
        if self.model is None:
            raise RuntimeError("Le modèle n'a pas été initialisé. Appelez prepare_data d'abord.")

        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                train_loader, optimizer, criterion, training=True, grad_clip=grad_clip
            )
            val_loss, val_acc = self._run_epoch(
                val_loader, None, criterion, training=False
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        print(f"Meilleur modèle chargé (val_loss={self.best_val_loss:.4f})")
        return self.history

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Modèle ou label encoder non initialisé.")

        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for features, labels in loader:
                logits, _ = self.model(features.to(self.device))
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        report_text = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.label_encoder.classes_),
            digits=3,
        )
        cm = confusion_matrix(all_labels, all_preds)

        print("Rapport de classification :")
        print(report_text)
        print("Matrice de confusion :")
        print(cm)

        accuracy = float(np.trace(cm) / np.maximum(1, cm.sum()))

        return {
            "classification_report": report_text,
            "confusion_matrix": cm,
            "labels": list(self.label_encoder.classes_),
            "accuracy": accuracy,
        }

    def save_training_artifacts(self, metrics: Dict[str, Any], output_dir: Path | str) -> None:
        """Enregistre les graphiques et rapports d'entraînement."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.history["train_loss"]:
            epochs = list(range(1, len(self.history["train_loss"]) + 1))
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(epochs, self.history["train_loss"], label="Train")
            axes[0].plot(epochs, self.history["val_loss"], label="Validation")
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

            axes[1].plot(epochs, self.history["train_acc"], label="Train")
            axes[1].plot(epochs, self.history["val_acc"], label="Validation")
            axes[1].set_title("Accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        report_text = metrics.get("classification_report")
        if report_text:
            (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

        cm = metrics.get("confusion_matrix")
        labels = metrics.get("labels")
        if cm is not None and labels:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar=False,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def save_model(self, output_dir: Path | str) -> Path:
        if self.model is None or self.scaler is None or self.label_encoder is None:
            raise RuntimeError("Le modèle doit être entraîné avant sauvegarde.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "feature_cols": self.feature_cols,
            "sequence_length": self.sequence_length,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "history": self.history,
        }

        ckpt_path = output_dir / "signal_generator.pth"
        torch.save(checkpoint, ckpt_path)
        print(f"Modèle sauvegardé: {ckpt_path}")
        return ckpt_path


def train_model_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement du Signal Generator (LSTM ou MLP)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/training_datasets/train_enhanced_with_historical.csv",
        help="Chemin vers le dataset d'entraînement.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "mlp"],
        default="lstm",
        help="Type de modèle à entraîner.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device à utiliser ("cpu", "cuda", "auto").',
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["random", "time", "symbol"],
        default="random",
        help="Méthode de split train/validation.",
    )
    parser.add_argument(
        "--val-symbols",
        nargs="*",
        default=None,
        help="Symboles réservés à la validation (utilisé si --split-strategy symbol).",
    )
    parser.add_argument(
        "--dataloader-timeout",
        type=float,
        default=0.0,
        help="Timeout (en secondes) pour les DataLoaders (augmenter si nécessaire).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Nombre de workers pour les DataLoaders (doit être >0 pour utiliser un timeout).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/signal_generator",
        help="Répertoire de sauvegarde du modèle.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Répertoire où enregistrer les graphiques et rapports (par défaut le dossier de sortie du modèle).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder le modèle après l'entraînement.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité."
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = ModelTrainer(model_type=args.model_type, device=args.device)
    train_loader, val_loader = trainer.prepare_data(
        args.csv_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        dataloader_timeout=args.dataloader_timeout,
        num_workers=args.num_workers,
        split_strategy=args.split_strategy,
        val_symbols=args.val_symbols,
    )

    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metrics = trainer.evaluate(val_loader)

    output_dir: Path | None = None
    if not args.no_save:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"{args.model_type}_{timestamp}"
        trainer.save_model(output_dir)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else output_dir
    if artifacts_dir is not None:
        trainer.save_training_artifacts(metrics, artifacts_dir)
        print(f"Graphiques et rapports enregistrés dans: {artifacts_dir}")


if __name__ == "__main__":
    train_model_cli()
