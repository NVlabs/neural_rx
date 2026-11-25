#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Compatibility wrapper for PyTorch/Sionna receiver training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_rx.cli import main as neural_rx_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_name", required=True)
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()

    config_path = Path(args.config_name)
    if not config_path.is_file():
        config_path = REPOSITORY_ROOT / "config" / args.config_name
    output = args.output or REPOSITORY_ROOT / "weights" / "runs" / f"{config_path.stem}.safetensors"
    device = "cpu" if args.debug or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    cli_args = [
        "train",
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--device",
        device,
    ]
    if args.steps is not None:
        cli_args.extend(["--max-steps", str(args.steps)])
    if args.batch_size is not None:
        cli_args.extend(["--batch-size", str(args.batch_size)])
    if args.init_checkpoint is not None:
        cli_args.extend(["--init-checkpoint", str(args.init_checkpoint)])
    return neural_rx_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
