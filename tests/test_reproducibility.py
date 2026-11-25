import torch
from sionna.phy import config


def test_seed_resets_both_torch_and_sionna():
    from neural_rx.reproducibility import seed_everything

    seed_everything(42)
    first = (torch.rand(4), torch.rand(4, generator=config.torch_rng('cpu')))
    seed_everything(42)
    second = (torch.rand(4), torch.rand(4, generator=config.torch_rng('cpu')))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
