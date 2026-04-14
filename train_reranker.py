"""Train the latent-intent reranker and prior bank."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.utils import logger


def main():
    parser = argparse.ArgumentParser(description="Train latent-intent reranker")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=24)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    data_loader = DataLoader()
    from src.latent_reranker import train_reranker

    result = train_reranker(
        data_loader,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negatives=args.negatives,
        force_rebuild=args.recompute,
    )
    logger.info(f"Training complete: {result}")


if __name__ == "__main__":
    main()
