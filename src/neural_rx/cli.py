# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Command-line entry points for the PyTorch receiver."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from neural_rx.config import load_experiment_config
from neural_rx.deployment import (
    AerialReceiver,
    aerial_example_inputs,
    export_aerial_onnx,
)
from neural_rx.evaluation import evaluate_checkpoint
from neural_rx.export import export_onnx
from neural_rx.model import NeuralReceiver, ReceiverOutput
from neural_rx.phy import SionnaPuschAdapter, sample_num_active_users
from neural_rx.reproducibility import seed_everything
from neural_rx.training import (
    apply_mcs_snr_offsets,
    load_checkpoint,
    receiver_loss,
    save_checkpoint,
)


def _train(args: argparse.Namespace) -> int:
    experiment = load_experiment_config(args.config)
    config = experiment.receiver
    device = torch.device(args.device)
    seed_everything(args.seed)
    model = NeuralReceiver(config).to(device)
    phases = list(experiment.training.phases)
    if args.phase_steps is not None:
        if len(args.phase_steps) != len(phases):
            raise ValueError(
                "--phase-steps must provide one value per configured phase"
            )
        phases = [
            replace(phase, steps=steps)
            for phase, steps in zip(phases, args.phase_steps)
        ]
    if args.batch_size is not None:
        phases = [replace(phase, batch_size=args.batch_size) for phase in phases]
    adapter = SionnaPuschAdapter.from_config(
        args.config,
        training_channel_type=phases[0].channel_type,
        device=str(device),
    )
    if args.init_checkpoint is not None:
        load_checkpoint(
            model,
            args.init_checkpoint,
            device=device,
            extra_tensors=adapter.checkpoint_tensors,
        )
    checkpoint_every = args.checkpoint_every or experiment.training.checkpoint_every
    adapter.set_transmitter_trainable(phases[0].train_tx)
    model_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(
        [*model_parameters, *adapter.transmitter_parameters],
        lr=phases[0].learning_rate,
    )

    def save(path: Path, steps: int, phase: int, loss: float) -> None:
        metadata = {
            "steps": steps,
            "phase": phase,
            "final_loss": loss,
            "seed": args.seed,
            "compile_model": args.compile_model,
            "amp": args.amp,
            "init_checkpoint": args.init_checkpoint and str(args.init_checkpoint),
        }
        save_checkpoint(
            model, path, config, metadata=metadata,
            extra_tensors=adapter.checkpoint_tensors,
        )

    final_loss = 0.0
    last_loss = None
    global_step = 0
    current_phase = 0
    model.train()
    # Compile only the forward callable. Optimizer parameters and serialized
    # state stay owned by the model, without wrapper-prefixed keys.
    # Full-graph failures propagate rather than silently changing execution.
    model_forward = (
        torch.compile(model.forward, fullgraph=True, dynamic=False)
        if args.compile_model else model
    )
    stop = False
    for current_phase, phase in enumerate(phases, start=1):
        if adapter.channel_type != phase.channel_type:
            transmitter_tensors = {
                name: tensor.detach().clone()
                for name, tensor in adapter.checkpoint_tensors.items()
            }
            adapter.set_transmitter_trainable(False)
            adapter = SionnaPuschAdapter.from_config(
                args.config,
                training_channel_type=phase.channel_type,
                device=str(device),
            )
            with torch.no_grad():
                for name, tensor in transmitter_tensors.items():
                    adapter.checkpoint_tensors[name].copy_(tensor)
            optimizer.param_groups[0]["params"] = [
                *model_parameters,
                *adapter.transmitter_parameters,
            ]
        adapter.set_transmitter_trainable(phase.train_tx)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = phase.learning_rate
        for _ in range(phase.steps):
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break
            num_active_users = int(
                sample_num_active_users(experiment.min_num_tx, experiment.max_num_tx)
            )
            minimum = phase.min_ebno_db[num_active_users - 1]
            maximum = phase.max_ebno_db[num_active_users - 1]
            if minimum == maximum:
                ebno_db = torch.full((phase.batch_size,), minimum, device=device)
            else:
                ebno_db = torch.empty((phase.batch_size,), device=device).uniform_(
                    minimum, maximum
                )
            if len(config.num_bits_per_symbol_by_mcs) == 1:
                mcs_indices = torch.zeros(
                    (phase.batch_size, experiment.max_num_tx),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                mcs_indices = torch.randint(
                    len(config.num_bits_per_symbol_by_mcs),
                    (phase.batch_size, experiment.max_num_tx),
                    device=device,
                )
            mcs_mask = torch.nn.functional.one_hot(
                mcs_indices, num_classes=len(config.num_bits_per_symbol_by_mcs)
            ).to(torch.float32)
            user_ranks = torch.rand(phase.batch_size, experiment.max_num_tx, device=device).argsort(dim=1)
            active_users = (user_ranks < num_active_users).float()
            ebno_db = apply_mcs_snr_offsets(
                ebno_db,
                mcs_mask,
                experiment.training.mcs_snr_db_offsets[num_active_users - 1],
                active_users,
            )
            batch = adapter.sample(
                phase.batch_size,
                ebno_db,
                num_active_users=num_active_users,
                active_users=active_users,
                mcs_mask=mcs_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                outputs = model_forward(
                    batch.received,
                    batch.pilot_features,
                    batch.active_users,
                    batch.initial_channel,
                    mcs_mask=batch.mcs_mask,
                    return_all_iterations=phase.multiloss,
                )
            if args.amp:
                outputs = tuple(
                    ReceiverOutput(output.llrs.float(), output.channel.float())
                    for output in outputs
                )
            losses = receiver_loss(
                outputs,
                batch.bit_targets,
                batch.active_users,
                data_mask=batch.data_mask,
                bit_mask=batch.bit_mask,
                channel_target=(
                    batch.channel_target if phase.channel_readout else None
                ),
                channel_weight=(phase.channel_weight if phase.channel_readout else 0.0),
            )
            losses.total.backward()
            optimizer.step()
            last_loss = losses.total.detach()
            global_step += 1
            if global_step in args.save_steps:
                save(
                    args.output.with_name(
                        f"{args.output.stem}_steps{global_step}{args.output.suffix}"
                    ),
                    global_step, current_phase, last_loss.item(),
                )
            if global_step % checkpoint_every == 0:
                final_loss = last_loss.item()
                save(args.output, global_step, current_phase, final_loss)
                print(
                    json.dumps(
                        {"loss": final_loss, "phase": current_phase, "steps": global_step},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if stop:
            break

    if last_loss is not None:
        final_loss = last_loss.item()
    save(args.output, global_step, current_phase, final_loss)
    print(
        json.dumps(
            {"checkpoint": str(args.output), "loss": final_loss, "steps": global_step},
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    results = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        args.ebno_db,
        batches=args.batches,
        batch_size=args.batch_size,
        device=args.device,
        target_block_errors=args.target_block_errors,
        target_bler=args.target_bler,
        early_stop=not args.no_early_stop,
        mcs_indices=args.mcs_indices,
        seed=args.seed,
    )
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def _export(args: argparse.Namespace) -> int:
    experiment = load_experiment_config(args.config)
    config = experiment.receiver
    model = NeuralReceiver(config).to(args.device)
    n_size_bwp = (
        experiment.evaluation.n_size_bwp if args.n_size_bwp is None else args.n_size_bwp
    )
    adapter = SionnaPuschAdapter.from_config(
        args.config, device=args.device, n_size_bwp_override=n_size_bwp
    )
    load_checkpoint(
        model,
        args.checkpoint,
        device=args.device,
        extra_tensors=adapter.checkpoint_tensors,
    )
    if args.profile == "aerial":
        if not config.initial_channel_estimate:
            raise ValueError("The Aerial profile requires an initial channel estimate")
        wrapper = AerialReceiver(
            model,
            adapter.num_users,
            adapter.pusch_configs[0].dmrs.num_cdm_groups_without_data,
        ).to(args.device)
        export_aerial_onnx(wrapper, aerial_example_inputs(adapter), args.output)
    else:
        batch = adapter.sample(1, 8.0)
        inputs = (
            batch.received,
            batch.pilot_features,
            batch.active_users,
        )
        if batch.initial_channel is not None:
            inputs = (*inputs, batch.initial_channel)
        if len(config.num_bits_per_symbol_by_mcs) > 1:
            inputs = (*inputs, batch.mcs_mask)
        export_onnx(model, inputs, args.output)
    print(
        json.dumps({"onnx": str(args.output), "profile": args.profile}, sort_keys=True)
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-rx")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train the configured Sionna experiment")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--phase-steps", type=int, nargs="+")
    train.add_argument("--checkpoint-every", type=int)
    train.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=(),
        help="also save <output>_steps<N> checkpoints at these steps",
    )
    train.add_argument(
        "--compile-model",
        action="store_true",
        help="compile only receiver forward with a static full graph; errors propagate",
    )
    train.add_argument(
        "--amp",
        action="store_true",
        help="run the receiver forward in bfloat16 autocast",
    )
    train.add_argument(
        "--init-checkpoint",
        type=Path,
        help="checkpoint used to initialize fine-tuning",
    )
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="evaluate BER/BLER with Sionna")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--ebno-db", nargs="+", type=float)
    evaluate.add_argument("--batches", type=int)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--target-block-errors", type=int, default=500)
    evaluate.add_argument("--target-bler", type=float, default=0.001)
    evaluate.add_argument("--no-early-stop", action="store_true")
    evaluate.add_argument(
        "--mcs-indices",
        nargs="+",
        type=int,
        help="configured MCS-array index for each evaluated user",
    )
    evaluate.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.set_defaults(handler=_evaluate)

    export = commands.add_parser("export", help="export a trained receiver to ONNX")
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    export.add_argument("--output", required=True, type=Path)
    export.add_argument(
        "--profile",
        choices=("native", "aerial"),
        default="native",
        help="ONNX input/output contract; aerial does not require TensorRT",
    )
    export.add_argument(
        "--n-size-bwp",
        type=int,
        help="exported grid size in PRBs; defaults to the evaluation n_size_bwp",
    )
    export.set_defaults(handler=_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
