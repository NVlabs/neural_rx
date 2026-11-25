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

"""Compatibility wrapper for PyTorch ONNX export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_rx.cli import main as neural_rx_main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_name", required=True)
    parser.add_argument("-weights", type=Path)
    parser.add_argument("-gpu", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = Path(args.config_name)
    if not config.is_file():
        config = REPOSITORY_ROOT / "config" / args.config_name
    checkpoint = (
        args.weights or REPOSITORY_ROOT / "weights" / "pretrained" / f"{config.stem}.safetensors"
    )
    output = args.output or REPOSITORY_ROOT / "onnx_models" / f"{config.stem}.onnx"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    return neural_rx_main(
        [
            "export",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--output",
            str(output),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
