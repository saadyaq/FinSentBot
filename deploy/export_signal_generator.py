#!/usr/bin/env python3
"""
Utility script to prepare the SignalGenerator checkpoint for deployment.

Usage examples:
  - Export the latest checkpoint found in models/signal_generator/*:
      python deploy/export_signal_generator.py --output-dir build/model

  - Export a specific checkpoint and upload it to S3:
      python deploy/export_signal_generator.py \
          --checkpoint models/signal_generator/mlp_20251020_204524/signal_generator.pth \
          --output-dir build/model --s3-uri s3://my-bucket/finsentbot/signal_generator/latest/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch

DEFAULT_CHECKPOINT_ROOT = Path("models") / "signal_generator"


def find_latest_checkpoint(base_dir: Path = DEFAULT_CHECKPOINT_ROOT) -> Path:
    """Retourne le checkpoint signal_generator.pth le plus récent."""
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Aucun dossier {base_dir} trouvé. Lancez un entraînement avant export."
        )

    candidates = sorted(base_dir.glob("*/signal_generator.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("Aucun fichier signal_generator.pth trouvé dans models/signal_generator.")
    return candidates[-1]


def load_metadata(checkpoint_path: Path) -> dict:
    """Extrait les métadonnées utiles pour le manifeste."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", {})
    return {
        "model_type": checkpoint.get("model_type", "unknown"),
        "feature_cols": checkpoint.get("feature_cols", []),
        "sequence_length": checkpoint.get("sequence_length", 1),
        "metrics": {
            "best_val_loss": checkpoint.get("best_val_loss"),
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
        },
    }


def write_manifest(output_dir: Path, checkpoint_path: Path, metadata: dict) -> None:
    manifest = {
        "model_file": checkpoint_path.name,
        "metadata": metadata,
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest écrit: {manifest_path}")


def upload_to_s3(local_path: Path, destination_uri: str) -> None:
    """Charge les fichiers de output_dir vers S3 (prefix)."""
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("boto3 est requis pour l'upload S3. Installez-le avec `pip install boto3`.") from exc

    if not destination_uri.startswith("s3://"):
        raise ValueError("L'URI S3 doit commencer par s3://bucket/prefix")

    bucket, _, prefix = destination_uri[5:].partition("/")
    s3_client = boto3.client("s3")

    print(f"Téléversement de {local_path} vers s3://{bucket}/{prefix}")
    for item in local_path.rglob("*"):
        if item.is_file():
            relative = item.relative_to(local_path)
            key = f"{prefix.rstrip('/')}/{relative.as_posix()}"
            print(f"  -> {key}")
            try:
                s3_client.upload_file(str(item), bucket, key)
            except (BotoCoreError, ClientError) as exc:  # pragma: no cover
                raise SystemExit(f"Échec upload {item}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prépare le modèle SignalGenerator pour le déploiement.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers signal_generator.pth (sinon le plus récent est utilisé).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Répertoire de sortie où copier le modèle et écrire le manifeste.",
    )
    parser.add_argument(
        "--s3-uri",
        type=str,
        default=None,
        help="Optionnel: destination S3 (ex: s3://bucket/path/) pour téléverser le package.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Nettoie le répertoire de sortie avant export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else find_latest_checkpoint()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    destination_ckpt = output_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, destination_ckpt)
    print(f"Checkpoint copié vers {destination_ckpt}")

    metadata = load_metadata(checkpoint_path)
    write_manifest(output_dir, destination_ckpt, metadata)

    if args.s3_uri:
        upload_to_s3(output_dir, args.s3_uri)
        print(f"Package uploadé sur {args.s3_uri}")

    print("Export terminé avec succès.")


if __name__ == "__main__":
    main()
