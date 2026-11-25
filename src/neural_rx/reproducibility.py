"""Seed independent simulation RNGs and describe the runtime environment."""

import hashlib
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sionna.phy import config


def seed_everything(seed: int) -> None:
    """Reset RNG streams; this does not promise cross-device determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config.seed = seed


def environment() -> dict[str, str | None]:
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'sionna': version('sionna'),
        'sionna-rt': version('sionna-rt'),
        'cuda': torch.version.cuda,
    }


def sha256(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
