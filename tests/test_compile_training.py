"""Opt-in forward compilation must preserve training and checkpoint ownership."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from neural_rx.cli import main
from neural_rx.model import NeuralReceiver


ROOT = Path(__file__).parents[1]


def training_arguments(checkpoint):
    return ["train", "--config", str(ROOT / "config/nrx_rt.cfg"),
            "--max-steps", "1", "--batch-size", "1", "--device", "cpu",
            "--output", str(checkpoint)]


def test_compiled_forward_preserves_eager_update_and_checkpoint(tmp_path, monkeypatch):
    original_compile = torch.compile
    compiled = []

    def compile_forward(forward, **kwargs):
        assert isinstance(forward.__self__, NeuralReceiver)
        assert forward.__name__ == "forward"
        assert kwargs == {"fullgraph": True, "dynamic": False}
        compiled.append(forward.__self__)
        # Real full-graph Dynamo capture; avoid expensive CPU Inductor codegen.
        captured_forward = original_compile(forward, backend="eager", **kwargs)

        def compare_forward(*args, **call_kwargs):
            expected = forward(*args, **call_kwargs)
            actual = captured_forward(*args, **call_kwargs)
            assert len(actual) == len(expected)
            for expected_output, actual_output in zip(expected, actual):
                torch.testing.assert_close(actual_output.llrs, expected_output.llrs)
                torch.testing.assert_close(actual_output.channel, expected_output.channel)
            return actual

        return compare_forward

    monkeypatch.setattr(torch, "compile", compile_forward)
    eager_path, compiled_path = tmp_path / "eager.safetensors", tmp_path / "compiled.safetensors"
    assert main(training_arguments(eager_path)) == 0
    assert not compiled  # Compilation remains strictly opt-in.
    assert main(training_arguments(compiled_path) + ["--compile-model"]) == 0
    assert len(compiled) == 1
    eager, captured = load_file(eager_path), load_file(compiled_path)
    assert eager.keys() == captured.keys()
    assert not any("_orig_mod" in key for key in captured)
    for key in eager:
        torch.testing.assert_close(captured[key], eager[key])
    eager_metadata = json.loads(eager_path.with_suffix(".json").read_text())
    captured_metadata = json.loads(compiled_path.with_suffix(".json").read_text())
    assert captured_metadata["final_loss"] == pytest.approx(eager_metadata["final_loss"])
    assert eager_metadata["compile_model"] is False
    assert captured_metadata["compile_model"] is True
    assert captured_metadata["steps"] == eager_metadata["steps"] == 1


def test_compile_failure_propagates_without_eager_fallback(tmp_path, monkeypatch):
    def compile_failure(forward, **kwargs):
        def failed_forward(*args, **call_kwargs):
            raise RuntimeError("deliberate compiler failure")
        return failed_forward

    monkeypatch.setattr(torch, "compile", compile_failure)
    checkpoint = tmp_path / "failed.safetensors"
    with pytest.raises(RuntimeError, match="deliberate compiler failure"):
        main(training_arguments(checkpoint) + ["--compile-model"])
    assert not checkpoint.exists()
