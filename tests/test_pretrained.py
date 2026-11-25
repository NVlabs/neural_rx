"""Checks for the shipped receiver checkpoints."""
from pathlib import Path

import pytest
import torch
from neural_rx.config import load_experiment_config
from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter
from neural_rx.training import load_checkpoint


ROOT = Path(__file__).parents[1]
MODELS = (
    "nrx_rt", "nrx_large", "e2e_rt", "e2e_large",
    "nrx_rt_qpsk", "nrx_rt_64qam", "nrx_rt_var_mcs", "nrx_large_qpsk",
    "nrx_large_64qam", "nrx_large_var_mcs", "nrx_large_var_mcs_64qam_masking",
    "nrx_site_specific", "nrx_site_specific_100k", "nrx_site_specific_large",
    "e2e_baseline", "nrx_oai",
)


@pytest.mark.parametrize("name", MODELS)
def test_shipped_checkpoint_loads_with_its_config(name):
    config = ROOT / "config" / f"{name}.cfg"
    model = NeuralReceiver(load_experiment_config(config).receiver)
    adapter = SionnaPuschAdapter.from_config(config, training_channel_type="UMi", device="cpu")
    metadata = load_checkpoint(model, ROOT / "weights" / "pretrained" / f"{name}.safetensors",
                               extra_tensors=adapter.checkpoint_tensors)
    assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
    expected = (100000 if name == "nrx_site_specific_100k" else
                200000 if name == "e2e_baseline" else
                7340000 if name == "nrx_rt_var_mcs" else
                2363000 if name == "nrx_large_var_mcs" else
                1000 if name.startswith("nrx_site_specific") else 2000000)
    assert metadata["steps"] == expected
