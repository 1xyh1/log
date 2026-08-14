import random

import numpy as np
import pytest
import torch

from mmod_qaf.train_loop import configure_reproducibility


def _rng_sample(seed: int):
    configure_reproducibility(seed, deterministic=False)
    return random.random(), np.random.random(), torch.rand(4)


def test_global_rngs_repeat_for_the_same_seed():
    first = _rng_sample(12345)
    second = _rng_sample(12345)
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_range_is_validated(seed):
    with pytest.raises(ValueError, match="seed must be"):
        configure_reproducibility(seed, deterministic=False)
